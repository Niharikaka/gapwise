# Gapwise — agentic AI course-design assistant (hosted demo)

ANL588 Applied Project, Karanam Niharika (B2511473), Singapore University of Social Sciences, 2026.

Streamlit app: `06_gapwise_app/gapwise_app.py`. It compares Singapore job-ad demand (MyCareersFuture) with course supply
(SkillsFuture and NUS/NTU/SMU/SUSS) and drafts evidence-tagged learning objectives for a training provider.

This repository is a slimmed hosting copy: job descriptions are kept only for AI-related postings (the only ones the
app reads). The full source code, raw data and analysis notebooks are in the project submission.

The optional DeepSeek rewrite reads `DEEPSEEK_API_KEY` from the app's Streamlit Secrets; no key is stored here.
