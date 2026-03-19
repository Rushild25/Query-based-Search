from __future__ import annotations


def root_layout(children: str, title: str = "Python Port") -> str:
    return f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f8fafc; }}
    .container {{ max-width: 1024px; margin: 0 auto; padding: 20px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 12px; background: white; margin-bottom: 16px; }}
    .card-content {{ padding: 16px; }}
    .button {{ border: 1px solid #1f2937; background: #111827; color: white; border-radius: 8px; padding: 8px 12px; cursor: pointer; }}
    .input {{ border: 1px solid #9ca3af; border-radius: 8px; padding: 8px; width: 100%; }}
    .row {{ display: flex; gap: 8px; align-items: center; }}
  </style>
</head>
<body>
  <div class=\"container\">{children}</div>
</body>
</html>
"""
