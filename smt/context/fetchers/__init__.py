"""Context data fetchers (live wiring). Each turns a free data source into the
`{direction, confidence}` context dicts the personas read. Network-free + graceful:
any failure returns None so the persona degrades to NEUTRAL (never fabricates a side).
"""
