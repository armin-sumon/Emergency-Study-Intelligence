def get_urgency(days):
    days = int(days)
    if days <= 1:
        return "CRITICAL"
    if days <= 3:
        return "HIGH"
    if days <= 7:
        return "MODERATE"
    return "LOW"
