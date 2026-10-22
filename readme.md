# Bhagavad Gita Knowledge Center & Life Counselor

An intelligent, Python-based spiritual counseling system that analyzes descriptive life problems (anxiety, grief, burnout, moral dilemmas, toxic conflicts, and self-doubt), semantically retrieves matching verses from the **Bhagavad Gita**, and synthesizes compassionate, actionable guidance rooted in Vedic philosophy.

Features a **built-in mobile web app**, **beta user feedback logging**, and a roadmap for **commercialization**.

---

## 🌟 Key Capabilities

1. **Built-in Mobile Web App**: A responsive, touch-friendly UI served directly at `http://<server-ip>:8000/` with "Add to Home Screen" PWA support.
2. **Beta User Feedback Loop**: One-tap thumbs up/down and user comment logging (`counseling_logs.jsonl`) to track verse matching accuracy during testing.
3. **Analytics Dashboard API**: Access live metrics via `GET /api/v1/analytics` to view query counts, helpfulness ratings, and satisfaction rates.
4. **Natural Language Problem Analysis**: Evaluates open-ended personal dilemmas, maps emotional states, and diagnoses root causes (*Kama*, *Krodha*, *Moha*, *Phala-asakti*).
5. **The Kurukshetra Mirror (Arjuna Analogy)**: Normalizes human struggle by connecting personal crises with Arjuna's crisis of despondency (*Arjuna Vishada*).
6. **Multi-Stage Semantic Retrieval**: Enhanced BM25 engine with conceptual query expansion and optional vector embedding hooks (`sentence-transformers`).
7. **Actionable 5-Pillar Synthesis**:
   - Emotional & Cognitive Mapping
   - The Kurukshetra Mirror (Arjuna's struggle)
   - Divine Prescriptions (Sanskrit Devanagari, IAST transliteration, translation, and commentary)
   - Practical Daily Sadhana Roadmap
   - Grounding Contemplation Mantra

---

## 👥 Phase 1: Testing with Real People (Beta Launch)

Before commercialization, you need feedback from 25–50 real users to validate accuracy and gather insights.

### 1. Host 24/7 for Free (No Need to Keep Your PC Running)

While running on your PC via Cloudflare Tunnel works for local demos, for beta testing with friends and community members, deploy to a **100% free 24/7 cloud platform**:

* **Option A: Render.com (Recommended Free Cloud Hosting)**
  1. Create a free account at [render.com](https://render.com).
  2. Connect your GitHub repository containing this project.
  3. Create a **New Web Service**:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn bhagavad_gita_knowledge_center:app --host 0.0.0.0 --port 10000`
  4. Render gives you a free, permanent HTTPS link (`https://gita-counselor.onrender.com`) to share with anyone!

* **Option B: Run on PC with Cloudflare Tunnel (Quickest for Today)**
  ```cmd
  cloudflared tunnel --url http://localhost:8000
  ```
  Share the generated `https://xyz.trycloudflare.com` URL with your testers.

### 2. View Live Beta Analytics

Check how people are using your app and whether the verses helped them:
* In your browser, open:
  ```text
  http://<server-url>/api/v1/analytics
  ```
* Returns live JSON stats:
  ```json
  {
    "total_queries": 42,
    "helpful_votes": 38,
    "unhelpful_votes": 4,
    "satisfaction_rate": 90.5,
    "recent_queries": [...]
  }
  ```

---

## 💼 Phase 2: Commercialization Roadmap (From Prototype to Business)

### 1. Revenue & Monetization Models

* **Freemium B2C Subscription**:
  - **Free Tier**: 3 life dilemmas per day, core verse database, text explanations.
  - **Premium Tier (₹149–₹299/month or $4.99/month)**:
    - Unlimited counseling sessions.
    - Soothing Sanskrit audio chanting with authentic pronunciation and meditation soundscapes.
    - Conversational deep-dives (asking follow-up questions about specific verses).
    - Daily morning contemplation notification + personal Sadhana habit tracker.
* **Gift of Wisdom (D2C Physical Products)**:
  - Curated, physical 54-card "Gita Contemplation Decks" or personalized annual journals branded with your platform.
* **Corporate & Institutional Wellness (B2B)**:
  - "Karma Yoga for Modern Burnout & Leadership": Offering stress-management and moral-clarity workshops for companies and universities.

### 2. Technical Stack for Scale (10,000+ Users)

* **Mobile App Framework**: Wrap your web app or build natively with **Flutter** or **React Native** to publish on the **Google Play Store** and **Apple App Store**.
* **Authentication**: Integrate **Supabase Auth** or **Firebase Auth** (Free tier covers up to 50,000 monthly active users for Google Sign-in and OTP).
* **Payment Gateways**:
  - India: **Razorpay** or **Cashfree** (supports UPI, RuPay, cards, and auto-recurring subscriptions).
  - Global: **Stripe**.
* **Database & Vector Search**:
  - Migrate from `gita_verses_corpus.json` to **PostgreSQL with pgvector** (on Supabase or Neon.tech) to handle millions of queries with sub-millisecond semantic search.

### 3. Ethical & Legal Guardrails

* **Mental Health Disclaimer**: Always present a prominent spiritual wellness notice clarifying that the platform provides philosophical contemplation and is not a clinical psychiatric crisis service.
* **Privacy Assurance**: Anonymize personal problems so users feel safe sharing their most intimate life challenges.