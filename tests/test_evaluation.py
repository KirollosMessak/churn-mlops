from churn.evaluation import best_threshold, campaign_profit

BIZ = {"offer_cost": 50, "saved_customer_value": 500, "offer_success_rate": 0.3}


def test_campaign_profit_math():
    y = [1, 1, 0, 0]
    p = [0.9, 0.2, 0.8, 0.1]
    # threshold 0.5 contacts customers 0 (churner) and 2 (not): 1 * 0.3 * 500 - 2 * 50 = 50
    assert campaign_profit(y, p, 0.5, BIZ) == 50
    assert campaign_profit(y, p, 0.95, BIZ) == 0  # nobody contacted


def test_best_threshold_prefers_contacting_only_likely_churners():
    y = [1] * 10 + [0] * 90
    p = [0.9] * 10 + [0.3] * 90
    t = best_threshold(y, p, BIZ)
    assert 0.3 < t <= 0.9
