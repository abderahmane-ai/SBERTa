# Prototype Collapse Fix

## Problem Analysis

Training run collapsed at step ~7000 with all tokens assigned to prototype 3 (p3=99-100%). Entropy dropped from 75% to near 0%.

## Root Causes (3 Independent Bugs)

### Bug 1: Diversity Loss Goes to Zero
**Issue:** Margin-based loss `relu(cos + 0.1)²` reaches zero when `cos < -0.1`
- By step 1000: prototypes at cos ≈ -0.10, diversity loss flatlines
- Remaining 99,000 steps have **zero geometric separation force**

**Fix:** Exponential repulsion `exp(cos) - exp(-1)`
- Always positive, always has gradient
- At cos=-0.10: loss=0.537 (constant repulsion)
- At cos=+0.80: loss=1.858 (strong repulsion against alignment)

### Bug 2: Balance Loss Too Weak
**Issue:** At collapse (p3=98%), balance contributes only ~0.06 vs smooth at ~3.7
- `lambda_balance=1.0` with floor at 6.25% (0.25/4)
- **60:1 disadvantage** against smooth pressure

**Fix:** 
- Increase `lambda_balance: 1.0 → 30.0` (30× stronger)
- Raise floor `balance_min_usage_factor: 0.25 → 0.5` (12.5% per prototype for K=4)

### Bug 3: Burn-in Too Short
**Issue:** Burn-in is 2000 steps but diversity loss already dead at step 1000
- Smooth kicks in at step 2100 with no geometric force remaining
- Collapse begins immediately

**Fix:**
- Extend `burnin_ratio: 0.02 → 0.05` (5000 steps instead of 2000)
- Extend `smooth_warmup_ratio: 0.10 → 0.15` (more gradual ramp)
- Reduce `lambda_smooth: 15.0 → 8.0` (until div/balance are proven effective)

## Config Changes

```python
# sberta/config.py
lambda_smooth: float = 8.0         # was 15.0
smooth_warmup_ratio: float = 0.15  # was 0.10
burnin_ratio: float = 0.05         # was 0.02
lambda_div: float = 5.0            # was 1.0 (with new loss formulation)
lambda_balance: float = 30.0       # was 1.0
balance_min_usage_factor: float = 0.5  # was 0.25
```

## Code Changes

### sberta/model.py - LanguagePrototypes.diversity_loss()
```python
def diversity_loss(self) -> torch.Tensor:
    """Exponential repulsion: exp(cos) - exp(-1)"""
    L_n = F.normalize(self.prototypes, dim=-1)
    cos = L_n @ L_n.T
    mask = torch.triu(torch.ones(self.K, self.K, device=cos.device), diagonal=1)
    repulsion = torch.exp(cos) - math.exp(-1.0)  # always positive
    return (repulsion * mask).sum() / (self.K * (self.K - 1) / 2.0)
```

## Expected Behavior After Fix

With λ_div=5.0 and the exp formulation:
- Normal operation (cos=-0.10): div contributes ~2.7
- Smooth at mid-training (w=0.5): contributes ~3.0
- Balance rescue signal: ~1.8 (30× stronger than before)

**Ratio: div + balance ≈ 4.5 vs smooth ≈ 3.0** — geometric forces now competitive

### Important: Balance Loss Only Fires Below 12.5%

With `balance_min_usage_factor=0.5` and K=4, `min_usage = 0.125` (12.5%).

**Balance behavior:**
- At 25% per prototype (healthy): `relu(0.125 - 0.25) = 0` — no signal (correct)
- At 20% (early drift): `relu(0.125 - 0.20) = 0` — still no signal
- At 10% (danger zone): `relu(0.125 - 0.10) = 0.025` — rescue fires

**This means:**
- **Primary defense against drift:** `loss_div` (fires continuously)
- **Secondary rescue:** `loss_balance` (fires only below 12.5%)
- **Monitoring threshold:** Any prototype < 15% indicates trouble (before balance fires)

The exp-repulsion diversity loss is your first line of defense and should prevent drift from starting.

## Verification Checklist

- [ ] Diversity loss stays > 0.5 throughout training (never flatlines)
- [ ] Entropy stays > 60% after step 10000
- [ ] No prototype drops below 15% after burn-in (warning threshold)
- [ ] No prototype dominates > 40% after burn-in
- [ ] Gradient norm stays < 50 (no explosions)
- [ ] No AMP skips after initial warmup

**Early warning signs:**
- Any prototype < 15% usage → drift starting (balance won't fire until 12.5%)
- Diversity loss < 0.3 → prototypes aligning (should stay ~0.5-0.6)
- Entropy < 70% → imbalance developing

## Additional Safety

Gradient clipping already in place at `max_grad_norm=1.0` (good default).

## Next Steps

1. Stop current training (already collapsed, unrecoverable)
2. Apply these fixes
3. Restart training from step 0
4. Monitor prototype usage and entropy closely for first 15k steps
