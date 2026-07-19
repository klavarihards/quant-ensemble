"""Black-Scholes European option pricing, Greeks, and a monthly ATM chain.

Pure Black-Scholes with a constant risk-free rate and no dividend yield
-- adequate for sizing a volatility-premium strategy's option legs, not
a production pricing engine.
"""

import math

import pandas as pd

DEFAULT_RISK_FREE_RATE = 0.05  # placeholder; the project has no r(t) series yet


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S, K, T, r, sigma):
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0:
        raise ValueError("T (years to expiry) must be positive")
    if sigma <= 0:
        raise ValueError("sigma (annualised vol) must be positive")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def price_option(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes price of a European option.

    S: spot price. K: strike. T: time to expiry in years.
    r: continuously-compounded annual risk-free rate.
    sigma: annualised volatility (e.g. VIX/100).
    option_type: "call" or "put".
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    if option_type == "put":
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def compute_greeks(S, K, T, r, sigma, option_type="call"):
    """Delta, gamma, theta (per calendar day), and vega (per 1 vol point).

    Gamma and vega are identical for calls and puts under Black-Scholes;
    delta and theta differ by option_type.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = _norm_pdf(d1)
    sqrt_t = math.sqrt(T)

    gamma = pdf_d1 / (S * sigma * sqrt_t)
    vega = S * pdf_d1 * sqrt_t / 100.0

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta_annual = -(S * pdf_d1 * sigma) / (2 * sqrt_t) - r * K * math.exp(-r * T) * _norm_cdf(d2)
    elif option_type == "put":
        delta = _norm_cdf(d1) - 1.0
        theta_annual = -(S * pdf_d1 * sigma) / (2 * sqrt_t) + r * K * math.exp(-r * T) * _norm_cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta_annual / 365.0,
        "vega": vega,
    }


def generate_monthly_chain(date, S, vix_level, r=DEFAULT_RISK_FREE_RATE, dte_days=30):
    """ATM straddle (call + put at the nearest-$1 strike) for one date.

    S: SPY spot on that date. vix_level: VIX index level (e.g. 16.5),
    used as sigma = vix_level / 100 -- VIX approximates 30-day ATM
    implied vol, so dte_days defaults to a monthly (~30-day) expiry.
    Returns a two-row DataFrame (call, put) with price and Greeks.
    """
    strike = round(S)
    T = dte_days / 365.0
    sigma = vix_level / 100.0

    rows = []
    for option_type in ("call", "put"):
        price = price_option(S, strike, T, r, sigma, option_type)
        greeks = compute_greeks(S, strike, T, r, sigma, option_type)
        rows.append(
            {
                "date": date,
                "spot": S,
                "strike": strike,
                "dte_days": dte_days,
                "implied_vol": vix_level,
                "option_type": option_type,
                "price": price,
                **greeks,
            }
        )
    return pd.DataFrame(rows)
