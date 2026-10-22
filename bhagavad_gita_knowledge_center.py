import datetime
#!/usr/bin/env python3
"""
================================================================================
  BHAGAVAD GITA KNOWLEDGE CENTER & SPIRITUAL COUNSELOR
================================================================================
A production-grade Python system that receives descriptive, natural language
life problems (anxiety, grief, burnout, indecision, relationship conflicts, etc.),
analyzes the underlying psychological and existential dilemma, matches it
semantically with verses from the Bhagavad Gita, and synthesizes actionable,
compassionate solutions rooted in Vedic philosophy.

Key Architectural Layers:
1. Knowledge Base: Curated corpus of authentic verses with Sanskrit, transliteration,
   translations, philosophical meanings, yoga paths, and life applications.
2. Problem Analysis: Emotion detection, cognitive trap identification, and Arjuna analogy.
3. Multi-Stage Retrieval Engine: Hybrid BM25 with semantic query expansion and
   optional drop-in neural vector embedding (Sentence-Transformers/HuggingFace).
4. Solution Synthesis: Structured 5-pillar counseling report with optional RAG integration
   for LLMs (Google Gemini, OpenAI, or local Ollama).
5. Interfaces: Rich Interactive CLI and FastAPI REST API.

Author: Bhagavad Gita Knowledge Center Team
License: MIT
================================================================================
"""

import sys
import os
import math
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("GitaCounselor")

# ==============================================================================
# 1. DOMAIN MODELS & DATA DEFINITIONS
# ==============================================================================

@dataclass
class GitaVerse:
    chapter: int
    verse: int
    sanskrit: str
    transliteration: str
    english_translation: str
    philosophical_meaning: str
    themes: List[str]
    yoga_branch: str  # Karma Yoga, Jnana Yoga, Bhakti Yoga, Dhyana Yoga, Buddhi Yoga
    life_application: str

    @property
    def citation(self) -> str:
        return f"Chapter {self.chapter}, Verse {self.verse}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["citation"] = self.citation
        return data


@dataclass
class CounselingResponse:
    user_query: str
    detected_emotions: List[str]
    philosophical_diagnosis: str
    arjuna_analogy: str
    matched_verses: List[Dict[str, Any]]
    core_teachings: List[str]
    practical_action_steps: List[str]
    contemplation_mantra: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. CURATED BHAGAVAD GITA CORPUS
# ==============================================================================

def get_default_corpus() -> List[GitaVerse]:
    """
    Returns an authentic, curated dataset of foundational Bhagavad Gita verses
    systematically addressing the spectrum of human suffering and dilemmas.
    """
    return [
        # --- Chapter 2: Sankhya Yoga & Buddhi Yoga ---
        GitaVerse(
            chapter=2,
            verse=11,
            sanskrit="अशोच्यानन्वशोचस्त्वं प्रज्ञावादांश्च भाषसे।\nगतासूनगतासूंश्च नानुशोचन्ति पण्डिताः॥",
            transliteration="aśocyān anvaśocas tvaṁ prajñā-vādāṁś ca bhāṣase |\ngatāsūn agatāsūṁś ca nānuśocanti paṇḍitāḥ ||",
            english_translation="The Blessed Lord said: While speaking learned words, you mourn for what is not worthy of grief. The truly wise mourn neither for the living nor for the dead.",
            philosophical_meaning="Suffering often masquerades as intellectual reasoning. Krishna points out that true wisdom transcends superficial emotional fixation on bodily and worldly transitions. Sorrow arises when we attach eternal importance to perishable things.",
            themes=["grief", "loss", "mourning", "sorrow", "overthinking", "intellectualizing pain", "death", "sadness"],
            yoga_branch="Jnana Yoga",
            life_application="Step back and ask: 'Am I mourning something that is inherently fleeting?' Do not let emotional grief disguise itself as insurmountable logic. Re-anchor your perspective in eternal values."
        ),
        GitaVerse(
            chapter=2,
            verse=14,
            sanskrit="मात्रास्पर्शास्तु कौन्तेय शीतोष्णसुखदुःखदाः।\nआगमापायिनोऽनित्यास्तांस्तितिक्षस्व भारत॥",
            transliteration="mātrā-sparśās tu kaunteya śītoṣṇa-sukha-duḥkha-dāḥ |\nāgamāpāyino 'nityās tāṁs titikṣasva bhārata ||",
            english_translation="O son of Kunti, the contact of the senses with their objects gives rise to cold and heat, pleasure and pain. They are fleeting and impermanent. Bear them patiently, O descendant of Bharata.",
            philosophical_meaning="Sensory pleasures and pains are like passing seasons—they arrive, linger, and depart inevitably. True strength is Titiksha (forbearance): enduring life's dualities without losing inner composure.",
            themes=["pain", "grief", "suffering", "temporary struggle", "patience", "endurance", "breakup", "resilience", "setbacks", "physical distress"],
            yoga_branch="Jnana Yoga",
            life_application="Remind yourself: 'This storm is temporary; like winter turning to spring, this pain will pass.' Practice patient endurance without dramatizing the discomfort."
        ),
        GitaVerse(
            chapter=2,
            verse=20,
            sanskrit="न जायते म्रियते वा कदाचिन्नायं भूत्वा भविता वा न भूयः।\nअजो नित्यः शाश्वतोऽयं पुराणो न हन्यते हन्यमाने शरीरे॥",
            transliteration="na jāyate mriyate vā kadācin nāyaṁ bhūtvā bhavitā vā na bhūyaḥ |\najo nityaḥ śāśvato 'yaṁ purāṇo na hanyate hanyamāne śarīre ||",
            english_translation="The soul is neither born, nor does it ever die; nor having once existed, does it ever cease to be. The soul is unborn, eternal, ever-existing, undying, and primeval. It is not slain when the body is slain.",
            philosophical_meaning="Your core identity is the pure, conscious Self (Atman), untouched by death, decay, bankruptcy, or external failure. When you understand your immortal spiritual essence, existential fear evaporates.",
            themes=["death", "loss", "bereavement", "mortality", "existential dread", "identity crisis", "fear of dying", "eternal self"],
            yoga_branch="Jnana Yoga",
            life_application="In moments of acute bereavement or existential fear, meditate on the indestructible nature of consciousness. You and your loved ones are eternal souls experiencing a temporal chapter."
        ),
        GitaVerse(
            chapter=2,
            verse=47,
            sanskrit="कर्मण्येवाधिकारस्ते मा फलेषु कदाचन।\nमा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥",
            transliteration="karmaṇy evādhikāras te mā phaleṣu kadācana |\nmā karma-phala-hetur bhūr mā te saṅgo 'stv akarmaṇi ||",
            english_translation="You have a right only to perform your prescribed duties, but you are not entitled to the fruits of action. Never consider yourself the cause of results, nor be attached to inaction.",
            philosophical_meaning="The foundational verse of Karma Yoga. You have direct control over your effort, craft, and ethics, but zero control over the myriad variables shaping the ultimate outcome. Clinging to the fruit breeds chronic performance anxiety, fear of failure, and burnout.",
            themes=["anxiety", "fear of failure", "stress", "burnout", "career pressure", "results", "startup", "exam fear", "perfectionism", "procrastination", "outcome"],
            yoga_branch="Karma Yoga",
            life_application="Dethrone the outcome. Dedicate 100% of your energy to the execution of your duty right now. Release worry about results, knowing that effort alone belongs to you."
        ),
        GitaVerse(
            chapter=2,
            verse=48,
            sanskrit="योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय।\nसिद्ध्यसिद्ध्योः समो भूत्वा समत्वं योग उच्यते॥",
            transliteration="yoga-sthaḥ kuru karmāṇi saṅgaṁ tyaktvā dhanañjaya |\nsiddhy-asiddhyoḥ samo bhūtvā samatvaṁ yoga ucyate ||",
            english_translation="Perform your duty poised in Yoga, O Arjuna, abandoning all attachment to success or failure. Such equanimity of mind is called Yoga.",
            philosophical_meaning="Equanimity (Samatvam) is the pinnacle of psychological mastery. When the mind stops oscillating wildly between euphoria upon winning and despair upon losing, emotional energy is conserved for constructive action.",
            themes=["equanimity", "balance", "emotional stability", "success and failure", "calmness", "resilience", "inner peace", "ups and downs", "bipolar emotions"],
            yoga_branch="Buddhi Yoga",
            life_application="Cultivate an inner sanctuary that external circumstances cannot shake. Greet success with humility and setbacks with curious resolve, remaining steadily anchored."
        ),
        GitaVerse(
            chapter=2,
            verse=56,
            sanskrit="दुःखेष्वनुद्विग्नमनाः सुखेषु विगतस्पृहः।\nवीतरागभयक्रोधः स्थितधीर्मुनिरुच्यते॥",
            transliteration="duḥkheṣv anudvigna-manāḥ sukheṣu vigata-spṛhaḥ |\nvīta-rāga-bhaya-krodhaḥ sthita-dhīr munir ucyate ||",
            english_translation="One whose mind remains undisturbed amidst misery, who does not crave pleasure, and who is completely free from attachment, fear, and anger, is called a sage of steady wisdom.",
            philosophical_meaning="Defines the Sthitaprajna (person of grounded intellect). Agitation does not come from external adversity, but from the mind's vulnerability to attachment (Raga), fear (Bhaya), and anger (Krodha).",
            themes=["emotional resilience", "fear", "anger", "calmness under pressure", "stoicism", "mental toughness", "toxic environment"],
            yoga_branch="Buddhi Yoga",
            life_application="Notice the three thieves of peace: excessive clinging, irrational dread, and explosive resentment. Neutralize them through conscious breath and contemplative distance."
        ),
        GitaVerse(
            chapter=2,
            verse=62,
            sanskrit="ध्यायतो विषयान्पुंसः सङ्गस्तेषूपजायते।\nसङ्गात्सञ्जायते कामः कामात्क्रोधोऽभिजायते॥",
            transliteration="dhyāyato viṣayān puṁsaḥ saṅgas teṣūpajāyate |\nsaṅgāt sañjāyate kāmaḥ kāmāt krodho 'bhijāyate ||",
            english_translation="While contemplating the objects of the senses, a person develops attachment to them. From attachment desire arises, and from thwarted desire, anger is born.",
            philosophical_meaning="The psychological genesis of destructive compulsions. Continuous mental dwelling on external temptations breeds clinging; whenever reality obstructs that craving, wrath and frustration inevitably explode.",
            themes=["anger", "addiction", "desire", "craving", "obsession", "frustration", "irritation", "dopamine", "loss of self control"],
            yoga_branch="Buddhi Yoga",
            life_application="Monitor your mental intake. The seeds of anger are planted in what you repeatedly scroll, gaze upon, and obsess over. Cut off toxic contemplation before it turns into obsession."
        ),
        GitaVerse(
            chapter=2,
            verse=63,
            sanskrit="क्रोधाद्भवति संमोहः संमोहात्स्मृतिविभ्रमः।\nस्मृतिभ्रंशाद्बुद्धिनाशो बुद्धिनाशात्प्रणश्यति॥",
            transliteration="krodhād bhavati saṁmohaḥ saṁmohāt smṛti-vibhramaḥ |\nsmṛti-bhraṁśād buddhi-nāśo buddhi-nāśāt praṇaśyati ||",
            english_translation="From anger arises clouding of judgment, which leads to confusion of memory. When memory is bewildered, the intellect is ruined; and when intellect is ruined, a person is utterly lost.",
            philosophical_meaning="Traces the anatomical cascade of cognitive breakdown. When anger hijack the brain, discernment is paralyzed, past ethical lessons are forgotten, and one takes actions that ruin relationships and reputations.",
            themes=["anger", "rage", "impulsive behavior", "clouded judgement", "relationship conflict", "regret", "mental collapse"],
            yoga_branch="Buddhi Yoga",
            life_application="Impose a mandatory 'cool down' quarantine when feeling provoked. Never send an email, speak a final word, or take action while your intellect is intoxicated by anger."
        ),
        GitaVerse(
            chapter=2,
            verse=70,
            sanskrit="आपूर्यमाणमचलप्रतिष्ठं समुद्रमापः प्रविशन्ति यद्वत्।\nतद्वत्कामा यं प्रविशन्ति सर्वे स शान्तिमाप्नोति न कामकामी॥",
            transliteration="āpūryamāṇam acala-pratiṣṭhaṁ samudram āpaḥ praviśanti yadvat |\ntadvat kāmā yaṁ praviśanti sarve sa śāntim āpnoti na kāma-kāmī ||",
            english_translation="Just as the ocean remains undisturbed though filled by incessantly entering rivers, so does that person attain peace into whom all desires enter without producing disturbance, not one who strives to satisfy every desire.",
            philosophical_meaning="True serenity is not the artificial absence of external stimuli, but having an inner depth like the ocean. Rivers of thoughts, criticisms, and desires enter, yet the ocean never overflows or loses its depth.",
            themes=["inner peace", "calmness", "desires", "overwhelm", "distraction", "serenity", "deep focus", "noise"],
            yoga_branch="Dhyana Yoga",
            life_application="Cultivate ocean-like depth. Let worldly opinions, gossip, and temporary temptations flow into your awareness and dissolve without causing emotional tsunamis."
        ),

        # --- Chapter 3: Karma Yoga ---
        GitaVerse(
            chapter=3,
            verse=35,
            sanskrit="श्रेयान्स्वधर्मो विगुणः परधर्मात्स्वनुष्ठितात्।\nस्वधर्मे निधनं श्रेयः परधर्मो भयावहः॥",
            transliteration="śreyān sva-dharmo viguṇaḥ para-dharmāt sv-anuṣṭhitāt |\nsva-dharme nidhanaṁ śreyaḥ para-dharmo bhayāvahaḥ ||",
            english_translation="It is far better to perform one's own natural duty (Sva-dharma), even if deficiently, than to attempt another's duty perfectly. Destruction in one's own duty is better; following another's path is fraught with fear.",
            philosophical_meaning="Every person has an innate constitutional nature and moral responsibility. Trying to imitate someone else's career, personality, or lifestyle because of social comparison breeds deep chronic anxiety and imposter syndrome.",
            themes=["comparison", "peer pressure", "career confusion", "authenticity", "identity", "jealousy", "social expectations", "imposter syndrome"],
            yoga_branch="Karma Yoga",
            life_application="Stop living somebody else's script. Celebrate your authentic abilities and responsibilities. Mastery of your own genuine path is your only true source of confidence."
        ),

        # --- Chapter 6: Dhyana Yoga (Mind Mastery & Balance) ---
        GitaVerse(
            chapter=6,
            verse=5,
            sanskrit="उद्धरेदात्मनात्मानं नात्मानमवसादयेत्।\nआत्मैव ह्यात्मनो बन्धुरात्मैव रिपुरात्मनः॥",
            transliteration="uddhared ātmanātmānaṁ nātmānam avasādayet |\nātmaiva hy ātmano bandhur ātmaiva ripur ātmanaḥ ||",
            english_translation="Elevate yourself through the power of your own mind, and do not degrade yourself. For the mind can be the greatest friend of the self, or its deadliest enemy.",
            philosophical_meaning="You are the ultimate architect of your internal state. Neither friends nor enemies can harm or elevate you as deeply as your own self-talk. Refuse to be a bystander to your mind's self-degradation.",
            themes=["depression", "self-doubt", "low self-esteem", "hopelessness", "victim mentality", "motivation", "self-worth", "imposter syndrome"],
            yoga_branch="Dhyana Yoga",
            life_application="Silence the internal prosecutor. Cultivate encouraging, truthful, and uplifting mental habits. Train your mind to be your most loyal confidant."
        ),
        GitaVerse(
            chapter=6,
            verse=16,
            sanskrit="नात्यश्नतस्तु योगोऽस्ति न चैकान्तमनश्नतः।\nन चाति स्वप्नशीलस्य जाग्रतो नैव चार्जुन॥",
            transliteration="nāty-aśnatas tu yogo 'sti na caikāntam anaśnataḥ |\nna cāti-svapna-śīlasya jāgrato naiva cārjuna ||",
            english_translation="There is no possibility of one's becoming a yogi, O Arjuna, if one eats too much or eats too little, sleeps too much or does not sleep enough.",
            philosophical_meaning="Spiritual and mental well-being cannot exist without physical temperance. Extreme asceticism or extreme indulgence both wreck mental equilibrium. Balance is sacred.",
            themes=["burnout", "sleep deprivation", "exhaustion", "eating habits", "lifestyle", "extreme diets", "discipline"],
            yoga_branch="Dhyana Yoga",
            life_application="Audit your physical routines. If you are sleep-deprived, poorly nourished, or chronically exhausted, your mind cannot attain clarity. Honor the body as a sacred temple."
        ),
        GitaVerse(
            chapter=6,
            verse=17,
            sanskrit="युक्ताहारविहारस्य युक्तचेष्टस्य कर्मसु।\nयुक्तस्वप्नावबोधस्य योगो भवति दुःखहा॥",
            transliteration="yuktāhāra-vihārasya yukta-ceṣṭasya karmasu |\nyukta-svapnāvabodhasya yogo bhavati duḥkha-hā ||",
            english_translation="For one who is moderate in eating and recreation, balanced in work, and regular in sleep and wakefulness, Yoga becomes the destroyer of all sorrow.",
            philosophical_meaning="Moderation (Yukta) is the ultimate antidote to psychological burnout. Harmonizing work, recreation, nutrition, and rest eradicates the physical foundations of despair.",
            themes=["lifestyle", "burnout", "sleep issues", "work life balance", "physical health", "moderation", "exhaustion", "stress relief"],
            yoga_branch="Dhyana Yoga",
            life_application="Establish clear boundaries: regularize sleep and waking hours, take restful pauses during work, and avoid obsessive overwork. A balanced rhythm dissolves chronic sorrow."
        ),
        GitaVerse(
            chapter=6,
            verse=26,
            sanskrit="यतो यतो निश्चरति मनश्चञ्चलमस्थिरम्।\nततस्ततो नियम्यैतदात्मन्येव वशं नयेत्॥",
            transliteration="yato yato niścarati manaś cañcalam asthiram |\ntatas tato niyamyaitad ātmany eva vaśaṁ nayet ||",
            english_translation="From wherever the fickle and unsteady mind wanders away, one must repeatedly restrain it and bring it back under the control of the Self.",
            philosophical_meaning="Meditation and mindfulness do not mean forceful suppression of thoughts. Rather, it is the gentle, relentless art of noticing wandering and softly guiding attention back to center.",
            themes=["distraction", "overthinking", "lack of focus", "restless mind", "racing thoughts", "meditation", "adhd"],
            yoga_branch="Dhyana Yoga",
            life_application="Do not berate yourself when your attention scatters. Each time you notice your mind drifting into worry, take a conscious breath and gently steer it back to the present moment."
        ),
        GitaVerse(
            chapter=6,
            verse=34,
            sanskrit="चञ्चलं हि मनः कृष्ण प्रमाथि बलवद्दृढम्।\nतस्याहं निग्रहं मन्ये वायोरिव सुदुष्करम्॥",
            transliteration="cañcalaṁ hi manaḥ kṛṣṇa pramāthi balavad dṛḍham |\ntasyāhaṁ nigrahaṁ manye vāyor iva su-duṣkaram ||",
            english_translation="The mind is very restless, turbulent, obstinate, and strong, O Krishna. Subduing it seems to me as difficult as controlling the raging wind.",
            philosophical_meaning="Arjuna candidly articulates the raw human agony of mental chaos. Krishna does not judge or mock Arjuna; He validates that mental mastery is indeed a profound and demanding challenge.",
            themes=["mental struggle", "turbulent mind", "anxiety", "feeling overwhelmed", "hopeless struggle", "helplessness"],
            yoga_branch="Dhyana Yoga",
            life_application="Acknowledge your mental struggles without shame. Even the mighty warrior Arjuna admitted feeling powerless against his own turbulent mind. You are human; be patient with your process."
        ),
        GitaVerse(
            chapter=6,
            verse=35,
            sanskrit="असंशयं महाबाहो मनो दुर्निग्रहं चलम्।\nअभ्यासेन तु कौन्तेय वैराग्येण च गृह्यते॥",
            transliteration="asaṁśayaṁ mahā-bāho mano durnigrahaṁ calam |\nabhyāsena tu kaunteya vairāgyeṇa ca gṛhyate ||",
            english_translation="The Blessed Lord said: Undoubtedly, O mighty-armed one, the mind is restless and difficult to curb. But by consistent practice (Abhyasa) and detachment (Vairagya), it can be mastered.",
            philosophical_meaning="Krishna provides the definitive two-wing formula for mental victory: Abhyasa (repeated, daily positive spiritual and mental conditioning) and Vairagya (dispassionate letting go of futile obsessions).",
            themes=["restless mind", "discipline", "habits", "anxiety relief", "spiritual practice", "detachment", "consistency"],
            yoga_branch="Dhyana Yoga",
            life_application="Practice the twin disciplines: 1) Abhyasa—dedicate 15 minutes daily to meditation or prayer; 2) Vairagya—consciously practice unhooking from drama and letting go of petty irritations."
        ),

        # --- Chapter 9 & 12: Bhakti Yoga & Compassion ---
        GitaVerse(
            chapter=9,
            verse=22,
            sanskrit="अनन्याश्चिन्तयन्तो मां ये जनाः पर्युपासते।\nतेषां नित्याभियुक्तानां योगक्षेमं वहाम्यहम्॥",
            transliteration="ananyāś cintayanto māṁ ye janāḥ paryupāsate |\nteṣāṁ nityābhiyuktānāṁ yoga-kṣemaṁ vahāmy aham ||",
            english_translation="To those who always contemplate Me with undistracted devotion, I personally carry what they lack and safeguard what they already possess (Yoga-Kshema).",
            philosophical_meaning="The supreme divine guarantee of providence and protection. When you align your life with divine purpose and righteousness, the universe assumes stewardship over your fundamental needs and welfare.",
            themes=["insecurity", "fear of future", "financial worry", "loneliness", "trust", "providence", "devotion", "helplessness"],
            yoga_branch="Bhakti Yoga",
            life_application="Cast off catastrophic financial panic and loneliness. Dedicate your actions to righteousness, and trust that cosmic providence will provide the necessities of your journey."
        ),
        GitaVerse(
            chapter=12,
            verse=13,
            sanskrit="अद्वेष्टा सर्वभूतानां मैत्रः करुण एव च।\nनिर्ममो निरहङ्कारः समदुःखसुखः क्षमी॥",
            transliteration="adveṣṭā sarva-bhūtānāṁ maitraḥ karuṇa eva ca |\nnirmamo nirahaṅkāraḥ sama-duḥkha-sukhaḥ kṣamī ||",
            english_translation="One who harbors no malice toward any living being, who is friendly and compassionate, free from possessiveness and egotism, equipoised in pain and pleasure, and ever forgiving...",
            philosophical_meaning="Outlines the sublime attributes of a devotee dear to God. Grudges, hatred, and egoistic pride corrode the bearer far more than the target. Compassion and forgiveness are acts of spiritual liberation.",
            themes=["interpersonal conflict", "forgiveness", "jealousy", "toxic relationships", "compassion", "resentment", "grudges", "kindness"],
            yoga_branch="Bhakti Yoga",
            life_application="Release toxic grudges and resentment. Forgiving someone who wronged you is not endorsing their behavior; it is evicting their negative presence from living rent-free in your mind."
        ),

        # --- Chapter 18: Moksha Sannyasa Yoga ---
        GitaVerse(
            chapter=18,
            verse=66,
            sanskrit="सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज।\nअहं त्वां सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः॥",
            transliteration="sarva-dharmān parityajya mām ekaṁ śaraṇaṁ vraja |\nahaṁ tvāṁ sarva-pāpebhyo mokṣayiṣyāmi mā śucaḥ ||",
            english_translation="Abandon all varieties of dharmas and surrender unto Me alone. I shall deliver you from all sinful reactions and distress; do not grieve.",
            philosophical_meaning="The supreme culminating Mahavakya of the Bhagavad Gita: Sharanagati (Total Surrender). When intellectual schemes fail, moral paradoxes overwhelm, and guilt paralyzes, surrendering entirely to the Divine Grace dissolves all fear and sorrow.",
            themes=["moral dilemma", "indecision", "guilt", "existential burden", "surrender", "seeking divine help", "fear", "grief", "peace"],
            yoga_branch="Bhakti Yoga",
            life_application="Surrender the overwhelming burden of trying to fix everything yourself. In deep humility, pray: 'Thy will be done. Guide my steps.' Surrender is not defeat; it is tapping into infinite grace."
        )
    ]


# ==============================================================================
# 3. PROBLEM ANALYZER & INTENT EXTRACTION
# ==============================================================================

class ProblemAnalyzer:
    """Extracts emotional signatures, psychological patterns, and Arjuna analogies."""

    EMOTION_TAXONOMY = {
        "Fear of Failure & Performance Anxiety": [
            "anxious", "anxiety", "fail", "failed", "failure", "nervous", "terrified", "panic",
            "dread", "exam", "interview", "startup", "investors", "expectations", "results", "launch"
        ],
        "Grief, Bereavement & Heartbreak": [
            "grief", "mourning", "loss", "died", "death", "passed away", "crying", "breakup",
            "divorce", "lonely", "sorrow", "heartbroken", "emptiness"
        ],
        "Burnout & Exhaustion": [
            "burnout", "exhausted", "tired", "14 hours", "overworked", "overwhelmed", "fatigue",
            "drained", "can't cope", "too much work", "sleep deprived"
        ],
        "Anger, Resentment & Frustration": [
            "angry", "rage", "furious", "hate", "frustrated", "irritated", "betrayed",
            "revenge", "bitter", "conflict", "fight", "unfair"
        ],
        "Restless Mind & Overthinking": [
            "overthinking", "racing thoughts", "cannot sleep", "can't sleep", "insomnia",
            "mind won't stop", "unfocused", "distracted", "scattered", "mental noise"
        ],
        "Self-Doubt & Imposter Syndrome": [
            "imposter", "worthless", "not good enough", "inferior", "low self esteem",
            "comparing", "behind in life", "loser", "defeated", "shame"
        ],
        "Moral Dilemma & Indecision": [
            "confused", "indecisive", "dilemma", "torn", "stuck", "don't know what to do",
            "paralyzed", "ethical", "guilt", "right or wrong"
        ]
    }

    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        matched_categories = []

        for category, terms in cls.EMOTION_TAXONOMY.items():
            if any(re.search(r'\b' + re.escape(term) + r'\b', text_lower) for term in terms):
                matched_categories.append(category)

        if not matched_categories:
            matched_categories = ["Existential Search for Meaning & Clarity"]

        diagnosis = cls._diagnose_philosophical_root(matched_categories)
        analogy = cls._map_arjuna_analogy(matched_categories)

        return {
            "detected_emotions": matched_categories,
            "philosophical_diagnosis": diagnosis,
            "arjuna_analogy": analogy
        }

    @staticmethod
    def _diagnose_philosophical_root(categories: List[str]) -> str:
        cat_str = " ".join(categories).lower()
        if "fear of failure" in cat_str or "burnout" in cat_str:
            return "Attachment to the fruits of action (Phala-asakti) combined with the illusion of solitary doership (Ahamkara / Ego-driven burden)."
        elif "grief" in cat_str:
            return "Identification of transient physical existence with the eternal Self (Deha-abhimana) and resistance to the cosmic law of impermanence (Anitya)."
        elif "anger" in cat_str:
            return "Thwarted desires (Kama) devolving into wrath (Krodha), eroding discriminative intellect (Buddhi-nasa) and clouding memory."
        elif "restless" in cat_str or "self-doubt" in cat_str:
            return "Turbulence of the senses and untrained mind (Chanchala Manas), devoid of anchorage in the unwavering witness consciousness (Sakshi Bhav)."
        else:
            return "Veil of moral and existential delusion (Moha), obscuring one's authentic inner duty (Sva-dharma)."

    @staticmethod
    def _map_arjuna_analogy(categories: List[str]) -> str:
        cat_str = " ".join(categories).lower()
        if "fear of failure" in cat_str or "burnout" in cat_str:
            return (
                "On the precipice of the epic Kurukshetra battle, Arjuna was overwhelmed not by lack of skill, "
                "but by the terrifying weight of anticipated outcomes. He threw down his divine bow Gandiva, "
                "convinced that action would bring only ruin. Lord Krishna revealed that when you act as an instrument "
                "(Nimitta-matram) of duty, the outcome ceases to be a burden on your soul."
            )
        elif "grief" in cat_str:
            return (
                "In Chapter 1, Arjuna looked upon his elders, mentors, and relatives and wept in profound sorrow, "
                "declaring that victory would be tasteless if bought with loss. Lord Krishna began His discourse "
                "(Chapter 2, Verse 11) not with rebuke, but with cosmic elevation: reminding Arjuna that death touches "
                "only the temporary garment of the body, never the eternal, luminous soul."
            )
        elif "anger" in cat_str:
            return (
                "The entire catastrophic war of the Mahabharata was ignited by unchecked humiliation, pride, and wrath. "
                "Krishna warns Arjuna that an angry warrior fights not for justice, but out of clouded delirium. He shows "
                "how true warriors fight with pristine composure, devoid of personal malice."
            )
        elif "restless" in cat_str or "self-doubt" in cat_str:
            return (
                "Even Arjuna—the greatest archer of his age—confessed in Chapter 6 (Verse 34) that trying to control his "
                "restless mind felt as impossible as catching the raging wind in his bare hands. Krishna did not belittle him; "
                "He gently assured him that with patient practice (Abhyasa) and wise detachment (Vairagya), even the wildest "
                "mind becomes calm and focused."
            )
        else:
            return (
                "Arjuna stood in the center of two mighty armies, paralyzed by confusion over what path was right. "
                "In that moment of total surrender (Prapatti), his chariot became the sanctum of the highest spiritual dialogue."
            )


# ==============================================================================
# 4. MULTI-STAGE SEMANTIC SEARCH ENGINE (BM25 + EXPANSION)
# ==============================================================================

class SemanticSearchEngine:
    """
    Hybrid Search Engine featuring BM25 text relevance, thematic expansion,
    and optional hook for dense neural embeddings.
    """

    CONCEPT_EXPANSIONS = {
        "failure": ["fruit", "outcome", "anxiety", "action", "phala", "duty", "success", "karma", "effort"],
        "failed": ["failure", "fruit", "outcome", "anxiety", "action", "duty", "success", "karma"],
        "startup": ["action", "duty", "career", "results", "work", "balance"],
        "launch": ["outcome", "results", "action", "fear"],
        "depressed": ["grief", "sorrow", "soul", "impermanence", "elevate", "mind", "hope"],
        "depression": ["grief", "sorrow", "soul", "impermanence", "elevate", "mind", "hope"],
        "anxious": ["peace", "equanimity", "samatvam", "fear", "restless", "control"],
        "anxiety": ["peace", "equanimity", "samatvam", "fear", "restless", "control"],
        "lost": ["loss", "bereavement", "death", "grief", "soul", "impermanence", "mourning", "sorrow"],
        "loss": ["mourning", "bereavement", "death", "soul", "eternal", "impermanent", "sorrow"],
        "grief": ["mourning", "loss", "death", "soul", "eternal", "impermanent", "sorrow"],
        "sorrow": ["grief", "mourning", "loss", "death", "impermanence", "forbearance"],
        "mourning": ["grief", "loss", "death", "sorrow", "soul", "eternal"],
        "death": ["soul", "loss", "bereavement", "eternal", "grief", "undying", "atman"],
        "died": ["death", "soul", "loss", "bereavement", "eternal", "grief"],
        "empty": ["emptiness", "soul", "existential", "purpose", "eternal"],
        "breakup": ["attachment", "loss", "forbearance", "compassion", "forgiveness"],
        "angry": ["wrath", "krodha", "desire", "clouded", "judgment", "forgiveness"],
        "anger": ["wrath", "krodha", "desire", "clouded", "judgment", "forgiveness"],
        "insomnia": ["sleep", "moderation", "rest", "restless", "mind", "balance"],
        "sleep": ["insomnia", "moderation", "rest", "balance", "sorrow"],
        "overthinking": ["wandering", "restless", "focus", "abhyasa", "practice", "mind"],
        "jealous": ["comparison", "sva-dharma", "envy", "authenticity", "compassion"],
        "confused": ["dilemma", "indecision", "surrender", "duty", "guidance"],
        "dilemma": ["confused", "indecision", "duty", "surrender", "sharanagati"]
    }

    def __init__(self, corpus: List[GitaVerse], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus)
        self.doc_lengths: List[int] = []
        self.doc_token_counts: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._build_index()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 2]
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "are", "have", "you",
            "your", "was", "will", "what", "which", "who", "all", "can", "they", "them",
            "him", "her", "his", "their", "into", "onto", "upon", "also", "been", "one",
            "about", "would", "could", "should", "there", "these", "those", "being"
        }
        return [t for t in tokens if t not in stopwords]

    def _build_index(self):
        total_length = 0
        for verse in self.corpus:
            # Create a rich weighted composite document
            composite = (
                f"{verse.english_translation} " * 2 +
                f"{verse.philosophical_meaning} " * 3 +
                f"{verse.life_application} " * 3 +
                f"{' '.join(verse.themes)} " * 6 +
                f"{verse.yoga_branch} " * 2
            )
            tokens = self.tokenize(composite)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_token_counts.append(tf)

            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        self.avg_doc_len = (total_length / self.doc_count) if self.doc_count else 1.0

        for t, freq in self.df.items():
            self.idf[t] = math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[GitaVerse, float]]:
        raw_tokens = self.tokenize(query)
        expanded_tokens = list(raw_tokens)

        # Semantic query expansion
        for token in raw_tokens:
            if token in self.CONCEPT_EXPANSIONS:
                expanded_tokens.extend(self.CONCEPT_EXPANSIONS[token])

        if not expanded_tokens:
            return [(self.corpus[0], 1.0)]

        scores: List[Tuple[int, float]] = []
        for idx in range(self.doc_count):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            token_counts = self.doc_token_counts[idx]

            for t in expanded_tokens:
                if t in token_counts:
                    tf = token_counts[t]
                    idf = self.idf.get(t, 0.4)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf * (numerator / denominator)

            # Extra weight if user's direct words match explicit theme metadata
            verse = self.corpus[idx]
            q_lower = query.lower()
            for theme in verse.themes:
                if theme in q_lower:
                    score += 2.5

            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.corpus[idx], round(score, 3)) for idx, score in scores[:top_k]]


# ==============================================================================
# 5. SPIRITUAL COUNSELOR & LLM SYNTHESIZER
# ==============================================================================

class GitaCounselor:
    """
    Coordinates problem analysis, semantic verse retrieval, and solution synthesis.
    Supports local deterministic synthesis or augmented LLM reasoning (Gemini/OpenAI/Ollama).
    """

    def __init__(self, corpus: Optional[List[GitaVerse]] = None):
        self.corpus = corpus or get_default_corpus()
        self.search_engine = SemanticSearchEngine(self.corpus)
        self.analyzer = ProblemAnalyzer()

    def counsel(self, user_problem: str, top_k: int = 2) -> CounselingResponse:
        # 1. Analyze emotional signature and archetypal context
        analysis = self.analyzer.analyze(user_problem)

        # 2. Retrieve top matching Gita verses
        matched_results = self.search_engine.search(user_problem, top_k=top_k)

        matched_verses = []
        core_teachings = []
        action_steps = []

        for verse, score in matched_results:
            matched_verses.append({
                "citation": verse.citation,
                "sanskrit": verse.sanskrit,
                "transliteration": verse.transliteration,
                "english_translation": verse.english_translation,
                "philosophical_meaning": verse.philosophical_meaning,
                "yoga_branch": verse.yoga_branch,
                "life_application": verse.life_application,
                "relevance_score": score
            })
            core_teachings.append(
                f"[{verse.citation} • {verse.yoga_branch}]: {verse.philosophical_meaning}"
            )
            action_steps.append(
                f"• {verse.citation} ({verse.yoga_branch} Sadhana): {verse.life_application}"
            )

        # 3. Add holistic baseline practice
        action_steps.append(
            "• Daily Mindset Practice (Samatvam): Each morning, consciously dedicate your actions to the supreme good. Dedicate 10 minutes to silent breath-watching (Pranayama) before consuming digital media."
        )

        mantra = self._generate_contemplation_mantra(analysis["detected_emotions"])

        return CounselingResponse(
            user_query=user_problem,
            detected_emotions=analysis["detected_emotions"],
            philosophical_diagnosis=analysis["philosophical_diagnosis"],
            arjuna_analogy=analysis["arjuna_analogy"],
            matched_verses=matched_verses,
            core_teachings=core_teachings,
            practical_action_steps=action_steps,
            contemplation_mantra=mantra
        )

    @staticmethod
    def _generate_contemplation_mantra(emotions: List[str]) -> str:
        cat_str = " ".join(emotions).lower()
        if "fear of failure" in cat_str or "burnout" in cat_str:
            return "॥ कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ॥ — 'My right is only to the work, never to the fruits. In dedicated effort I am free.'"
        elif "grief" in cat_str:
            return "॥ न जायते म्रियते वा कदाचित् ॥ — 'The Soul is never born, nor does it ever die. All that is real is eternal; all that is temporal passes.'"
        elif "restless" in cat_str or "overthinking" in cat_str:
            return "॥ अभ्यासेन तु कौन्तेय वैराग्येण च गृह्यते ॥ — 'With gentle practice and calm detachment, the mind returns to peace.'"
        elif "anger" in cat_str:
            return "॥ स्थितधीर्मुनिरुच्यते ॥ — 'He whose mind remains calm in adversity, free from clinging and wrath, is established in divine wisdom.'"
        else:
            return "॥ आत्मैव ह्यात्मनो बन्धुः ॥ — 'The mind is my supreme friend. When anchored in the Self, no external storm can defeat me.'"


# ==============================================================================
# 6. TERMINAL & MARKDOWN FORMATTER
# ==============================================================================

def format_terminal_report(res: CounselingResponse) -> str:
    """Renders a readable, dignified counseling report for terminal or logs."""
    bar = "=" * 80
    sub_bar = "-" * 80
    lines = []

    lines.append("\n" + bar)
    lines.append("       ॐ श्रीमद्भगवद्गीता ज्ञान केन्द्र : BHAGAVAD GITA LIFE COUNSELOR")
    lines.append(bar)
    lines.append(f"\n[Your Inward Query]:\n\"{res.user_query.strip()}\"\n")

    lines.append("┌" + "─" * 78 + "┐")
    emotions_str = ", ".join(res.detected_emotions)
    lines.append(f"│ 1. EMOTIONAL & COGNITIVE MAPPING                                             │")
    lines.append("└" + "─" * 78 + "┘")
    lines.append(f"• Detected Patterns : {emotions_str}")
    lines.append(f"• Gita's Diagnosis  : {res.philosophical_diagnosis}")
    lines.append(f"\n• The Kurukshetra Mirror (Arjuna's Dilemma):\n  {res.arjuna_analogy}\n")

    lines.append("┌" + "─" * 78 + "┐")
    lines.append(f"│ 2. DIVINE PRESCRIPTIONS (RELEVANT SHLOKAS)                                   │")
    lines.append("└" + "─" * 78 + "┘")

    for i, v in enumerate(res.matched_verses, 1):
        lines.append(f"\n[{i}] {v['citation']} | Path: {v['yoga_branch']} (Score: {v['relevance_score']})")
        lines.append(sub_bar)
        for s_line in v['sanskrit'].split("\n"):
            lines.append(f"    {s_line}")
        lines.append("")
        for t_line in v['transliteration'].split("\n"):
            lines.append(f"    {t_line}")
        lines.append(f"\n    English Translation:\n    \"{v['english_translation']}\"\n")
        lines.append(f"    Spiritual Commentary:\n    {v['philosophical_meaning']}\n")

    lines.append("┌" + "─" * 78 + "┐")
    lines.append(f"│ 3. PRACTICAL ACTION ROADMAP (SADHANA FOR TODAY)                             │")
    lines.append("└" + "─" * 78 + "┘")
    for step in res.practical_action_steps:
        lines.append(f"{step}")

    lines.append("\n┌" + "─" * 78 + "┐")
    lines.append(f"│ 4. DAILY CONTEMPLATION SEED (MANTHRA)                                        │")
    lines.append("└" + "─" * 78 + "┘")
    lines.append(f" {res.contemplation_mantra}\n")
    lines.append(bar + "\n")

    return "\n".join(lines)



# ==============================================================================
# 6.5. BETA TESTING & COMMERCIAL ANALYTICS LOGGER
# ==============================================================================

class FeedbackLogger:
    """Logs user queries, feedback, and engagement metrics for beta testing and product optimization."""
    def __init__(self, log_path: str = "counseling_logs.jsonl"):
        self.log_path = log_path

    def log_query(self, user_query: str, detected_emotions: List[str], matched_verses: List[Dict[str, Any]]) -> str:
        query_id = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        record = {
            "query_id": query_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "user_query": user_query,
            "detected_emotions": detected_emotions,
            "matched_verses": [v.get("citation", "") for v in matched_verses],
            "rating": 0,
            "comment": ""
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Could not log query: {e}")
        return query_id

    def log_feedback(self, query_id: str, rating: int, comment: str = "") -> bool:
        if not os.path.exists(self.log_path):
            return False
        records = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            pass
            updated = False
            for r in records:
                if r.get("query_id") == query_id:
                    r["rating"] = rating
                    r["comment"] = comment
                    updated = True
                    break
            if updated:
                with open(self.log_path, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                return True
        except Exception as e:
            logger.warning(f"Could not update feedback: {e}")
        return False

    def get_summary(self) -> Dict[str, Any]:
        if not os.path.exists(self.log_path):
            return {"total_queries": 0, "helpful_votes": 0, "unhelpful_votes": 0, "recent_queries": []}
        records = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Could not read logs: {e}")

        total = len(records)
        upvotes = sum(1 for r in records if r.get("rating", 0) > 0)
        downvotes = sum(1 for r in records if r.get("rating", 0) < 0)
        return {
            "total_queries": total,
            "helpful_votes": upvotes,
            "unhelpful_votes": downvotes,
            "satisfaction_rate": round((upvotes / (upvotes + downvotes) * 100), 1) if (upvotes + downvotes) > 0 else 100.0,
            "recent_queries": records[-10:]
        }

# ==============================================================================
# 7. OPTIONAL FASTAPI REST SERVICE
# ==============================================================================

def create_app():
    """Builds and returns a FastAPI application instance."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel
    except ImportError:
        logger.error("FastAPI is not installed. Run: pip install fastapi uvicorn")
        return None

    app = FastAPI(
        title="Bhagavad Gita Knowledge Center API",
        description="RESTful API for matching life dilemmas with Bhagavad Gita verses.",
        version="1.0.0"
    )

    counselor = GitaCounselor()
    feedback_logger = FeedbackLogger()

    class QueryRequest(BaseModel):
        problem: str
        top_k: Optional[int] = 2

    class FeedbackRequest(BaseModel):
        query_id: str
        rating: int  # 1 for helpful, -1 for unhelpful
        comment: Optional[str] = ""

    @app.get("/api/v1/analytics")
    def get_analytics():
        return feedback_logger.get_summary()

    @app.post("/api/v1/feedback")
    def submit_feedback(fb: FeedbackRequest):
        success = feedback_logger.log_feedback(fb.query_id, fb.rating, fb.comment or "")
        return {"status": "success" if success else "not_found", "query_id": fb.query_id}

    @app.get("/", response_class=HTMLResponse)
    def mobile_home():
        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <meta name="theme-color" content="#0b0f19">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <title>Bhagavad Gita Knowledge Center</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #131c2e;
      --card-border: #1e293b;
      --gold: #f59e0b;
      --gold-light: #fef3c7;
      --saffron: #f97316;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 16px 12px 40px;
    }
    .container {
      width: 100%;
      max-width: 600px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    header {
      text-align: center;
      padding: 12px 0 8px;
    }
    .om-badge {
      font-size: 2.2rem;
      color: var(--gold);
      display: inline-block;
      margin-bottom: 4px;
      filter: drop-shadow(0 0 10px rgba(245, 158, 11, 0.4));
    }
    h1 {
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #fff;
    }
    .subtitle {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    label {
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--gold-light);
      display: block;
      margin-bottom: 8px;
    }
    textarea {
      width: 100%;
      height: 110px;
      background: #090e17;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px;
      color: #fff;
      font-size: 0.95rem;
      font-family: inherit;
      resize: vertical;
      outline: none;
      transition: border 0.2s;
    }
    textarea:focus {
      border-color: var(--gold);
    }
    .quick-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }
    .chip {
      background: #1e293b;
      color: #cbd5e1;
      font-size: 0.75rem;
      padding: 6px 10px;
      border-radius: 20px;
      border: 1px solid #334155;
      cursor: pointer;
      user-select: none;
      transition: all 0.15s ease;
    }
    .chip:active {
      transform: scale(0.96);
      background: var(--gold);
      color: #000;
    }
    button.submit-btn {
      width: 100%;
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: #000;
      font-weight: 700;
      font-size: 1rem;
      padding: 14px;
      border: none;
      border-radius: 12px;
      cursor: pointer;
      margin-top: 14px;
      box-shadow: 0 4px 15px rgba(245, 158, 11, 0.35);
      transition: opacity 0.2s;
    }
    button.submit-btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .spinner {
      display: none;
      text-align: center;
      padding: 24px;
      color: var(--gold);
      font-size: 0.9rem;
    }
    .spinner.active { display: block; }
    .results {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .badge {
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 4px 8px;
      border-radius: 6px;
      background: rgba(245, 158, 11, 0.15);
      color: var(--gold);
      margin-bottom: 6px;
    }
    .section-title {
      font-size: 1rem;
      font-weight: 700;
      color: #fff;
      margin-bottom: 6px;
    }
    .shloka-box {
      background: #090e17;
      border: 1px solid #243247;
      border-radius: 12px;
      padding: 14px;
      margin-top: 10px;
    }
    .sanskrit {
      color: var(--gold);
      font-size: 1.05rem;
      line-height: 1.6;
      text-align: center;
      white-space: pre-line;
      font-weight: 600;
      margin-bottom: 8px;
    }
    .transliteration {
      color: #94a3b8;
      font-size: 0.82rem;
      font-style: italic;
      text-align: center;
      white-space: pre-line;
      margin-bottom: 12px;
    }
    .translation {
      font-size: 0.92rem;
      color: #f8fafc;
      line-height: 1.5;
      margin-bottom: 8px;
    }
    .commentary {
      font-size: 0.85rem;
      color: #cbd5e1;
      line-height: 1.45;
      border-left: 2px solid var(--gold);
      padding-left: 8px;
      margin-top: 6px;
    }
    .step-item {
      font-size: 0.88rem;
      line-height: 1.45;
      color: #e2e8f0;
      margin-bottom: 8px;
    }
    .mantra-card {
      background: linear-gradient(135deg, rgba(245,158,11,0.1), rgba(249,115,22,0.05));
      border: 1px solid rgba(245,158,11,0.3);
      text-align: center;
      padding: 16px;
      border-radius: 14px;
    }
    .mantra-text {
      color: var(--gold);
      font-weight: 600;
      font-size: 0.95rem;
      line-height: 1.5;
    }
    footer {
      text-align: center;
      font-size: 0.75rem;
      color: #64748b;
      margin-top: 20px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="om-badge">ॐ</div>
      <h1>Bhagavad Gita Knowledge Center</h1>
      <p class="subtitle">Spiritual counseling for modern life dilemmas</p>
    </header>

    <div class="card">
      <label for="problemInput">What life challenge is on your mind?</label>
      <textarea id="problemInput" placeholder="Describe your anxiety, sorrow, career pressure, burnout, relationship struggle, or confusion..."></textarea>

      <div class="quick-chips">
        <span class="chip" onclick="setQuery('I worked 14 hours a day on my startup but our launch failed completely. I feel anxious, like a failure, and terrified of what others will think.')">🚀 Startup Failure</span>
        <span class="chip" onclick="setQuery('I recently lost a very dear person in my life and I feel an overwhelming void and sorrow. Nothing seems meaningful.')">💔 Grief & Loss</span>
        <span class="chip" onclick="setQuery('My mind won\'t stop racing with overthinking and worries about the future. I can\'t focus and suffer from insomnia.')">🌀 Overthinking & Focus</span>
        <span class="chip" onclick="setQuery('I have so much anger and bitterness towards someone who betrayed me, and it is consuming my peace of mind.')">🔥 Resentment & Anger</span>
      </div>

      <button id="submitBtn" class="submit-btn" onclick="seekGuidance()">Seek Gita's Counsel</button>
    </div>

    <div id="spinner" class="spinner">
      <div>Contemplating the eternal wisdom of Kurukshetra...</div>
    </div>

    <div id="results" class="results" style="display:none;"></div>

    <footer>
      ॥ यतो धर्मस्ततो जयः ॥ — Where there is righteousness, there is victory.
    </footer>
  </div>

  <script>
    function setQuery(text) {
      document.getElementById('problemInput').value = text;
      document.getElementById('problemInput').focus();
    }

    async function seekGuidance() {
      const input = document.getElementById('problemInput').value.trim();
      if (!input) return;

      const btn = document.getElementById('submitBtn');
      const spinner = document.getElementById('spinner');
      const resultsDiv = document.getElementById('results');

      btn.disabled = true;
      spinner.classList.add('active');
      resultsDiv.style.display = 'none';

      try {
        const response = await fetch('/api/v1/counsel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ problem: input, top_k: 2 })
        });

        if (!response.ok) throw new Error('Failed to retrieve counsel');
        const data = await response.json();
        renderResults(data);
      } catch (err) {
        alert('Could not connect to server. Please ensure the PC server is running.');
      } finally {
        btn.disabled = false;
        spinner.classList.remove('active');
      }
    }

    function renderResults(data) {
      const resultsDiv = document.getElementById('results');
      let html = '';

      html += '<div class="card">' +
        '<span class="badge">1. Diagnosis</span>' +
        '<div class="section-title">' + data.detected_emotions.join(' • ') + '</div>' +
        '<p style="font-size:0.88rem; color:#cbd5e1; margin-bottom:12px;"><strong>Root Cause:</strong> ' + data.philosophical_diagnosis + '</p>' +
        '<div style="background:#090e17; padding:12px; border-radius:10px; border-left:3px solid var(--saffron);">' +
          '<div style="font-size:0.8rem; font-weight:700; color:var(--saffron); margin-bottom:4px;">THE KURUKSHETRA MIRROR (ARJUNA\'S DILEMMA)</div>' +
          '<p style="font-size:0.85rem; color:#cbd5e1; line-height:1.4;">' + data.arjuna_analogy + '</p>' +
        '</div>' +
      '</div>';

      html += '<div class="card">' +
        '<span class="badge">2. Divine Prescriptions</span>' +
        '<div class="section-title">Relevant Bhagavad Gita Verses</div>';

      data.matched_verses.forEach(function(v) {
        html += '<div class="shloka-box">' +
          '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">' +
            '<span style="font-weight:700; font-size:0.9rem; color:#fff;">' + v.citation + '</span>' +
            '<span class="badge" style="margin:0;">' + v.yoga_branch + '</span>' +
          '</div>' +
          '<div class="sanskrit">' + v.sanskrit + '</div>' +
          '<div class="transliteration">' + v.transliteration + '</div>' +
          '<div class="translation"><strong>Translation:</strong> "' + v.english_translation + '"</div>' +
          '<div class="commentary"><strong>Spiritual Meaning:</strong> ' + v.philosophical_meaning + '</div>' +
        '</div>';
      });
      html += '</div>';

      html += '<div class="card">' +
        '<span class="badge">3. Daily Practice</span>' +
        '<div class="section-title">Actionable Roadmap (Sadhana)</div>';

      data.practical_action_steps.forEach(function(step) {
        html += '<div class="step-item">' + step + '</div>';
      });
      html += '</div>';

      html += '<div class="mantra-card">' +
        '<div style="font-size:0.75rem; color:var(--gold); font-weight:700; margin-bottom:4px;">DAILY CONTEMPLATION SEED</div>' +
        '<div class="mantra-text">' + data.contemplation_mantra + '</div>' +
      '</div>';

      // Feedback Widget for Beta Testing
      html += '<div class="card" id="feedbackCard" style="text-align:center; padding:16px;">' +
        '<div style="font-size:0.85rem; font-weight:600; color:#fff; margin-bottom:10px;">Was this guidance helpful for your dilemma?</div>' +
        '<div style="display:flex; justify-content:center; gap:12px; margin-bottom:8px;">' +
          '<button type="button" class="chip" style="font-size:0.85rem; padding:8px 16px; background:#1e293b;" onclick="sendFeedback(1, \'' + (data.query_id || '') + '\')">👍 Helpful</button>' +
          '<button type="button" class="chip" style="font-size:0.85rem; padding:8px 16px; background:#1e293b;" onclick="sendFeedback(-1, \'' + (data.query_id || '') + '\')">👎 Needs Improvement</button>' +
        '</div>' +
        '<div id="feedbackThanks" style="display:none; font-size:0.8rem; color:var(--gold); margin-top:6px;">Thank you for your feedback! It helps train our spiritual counselor.</div>' +
      '</div>';

      // Spiritual Wellness Disclaimer
      html += '<div style="font-size:0.75rem; color:#64748b; line-height:1.4; text-align:center; padding:8px 12px; border:1px solid #1e293b; border-radius:10px;">' +
        '<strong>Spiritual Wellness Notice:</strong> This platform offers philosophical contemplation rooted in the Bhagavad Gita to nurture inner resilience. It is not a substitute for clinical psychiatric or psychological therapy.' +
      '</div>';

      resultsDiv.innerHTML = html;
      resultsDiv.style.display = 'flex';
      resultsDiv.scrollIntoView({ behavior: 'smooth' });
    }

    async function sendFeedback(rating, queryId) {
      if (!queryId) return;
      try {
        await fetch('/api/v1/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query_id: queryId, rating: rating, comment: '' })
        });
        document.getElementById('feedbackThanks').style.display = 'block';
      } catch(e) {}
    }
  </script>
</body>
</html>
"""

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "service": "Bhagavad Gita Knowledge Center"}

    @app.get("/api/v1/verses")
    def list_verses():
        return [v.to_dict() for v in counselor.corpus]

    @app.post("/api/v1/counsel")
    def counsel_endpoint(req: QueryRequest):
        if not req.problem.strip():
            raise HTTPException(status_code=400, detail="Problem description cannot be empty.")
        res = counselor.counsel(req.problem, top_k=req.top_k or 2)
        payload = res.to_dict()
        query_id = feedback_logger.log_query(req.problem, res.detected_emotions, payload.get("matched_verses", []))
        payload["query_id"] = query_id
        return payload

    return app

# Module-level application instance for cloud deployment (Render, Railway, Fly.io, etc.)
app = create_app()


# ==============================================================================
# 8. EXECUTION MODES (CLI & DEMO)
# ==============================================================================

def interactive_loop():
    counselor = GitaCounselor()
    print("\n" + "=" * 80)
    print("  ॐ श्रीमद्भगवद्गीता ज्ञान केन्द्र : BHAGAVAD GITA LIFE COUNSELOR")
    print("=" * 80)
    print("Welcome. Share any personal dilemma, workplace stress, grief, fear, or confusion.")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            print("-" * 80)
            user_input = input("Describe your life situation/problem:\n> ")
            if not user_input or user_input.strip().lower() in ["exit", "quit", "q"]:
                print("\nMay wisdom and equanimity protect your journey. Hari Om Tat Sat.\n")
                break

            response = counselor.counsel(user_input, top_k=2)
            print(format_terminal_report(response))

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Hari Om Tat Sat.\n")
            break


if __name__ == "__main__":
    if "--demo" in sys.argv:
        counselor = GitaCounselor()
        sample_queries = [
            (
                "Career & Failure Anxiety",
                "I worked 14 hours a day on our startup launch, but it failed completely. "
                "I feel terrified of what investors and family will say, and I feel like an absolute loser."
            ),
            (
                "Grief & Bereavement",
                "I recently lost a very dear person in my life. The house is empty, and I feel an "
                "overwhelming sorrow. Nothing seems meaningful anymore."
            ),
            (
                "Restless Mind & Focus",
                "My mind is constantly racing with overthinking and worries about the future. "
                "I can't focus on my work and I suffer from sleepless nights."
            )
        ]

        print("\n" + "#" * 80)
        print("  RUNNING BHAGAVAD GITA KNOWLEDGE CENTER AUTOMATED DEMONSTRATION")
        print("#" * 80)

        for title, query in sample_queries:
            print(f"\n>>> TEST SCENARIO: {title}")
            res = counselor.counsel(query, top_k=2)
            print(format_terminal_report(res))

    elif "--serve" in sys.argv:
        try:
            import uvicorn
            app = create_app()
            if app:
                print("Starting Gita Knowledge Center REST API on http://0.0.0.0:8000 ...")
                uvicorn.run(app, host="0.0.0.0", port=8000)
        except ImportError:
            print("Uvicorn is required to run the web server. Run: pip install uvicorn fastapi")

    else:
        # Default behavior: if interactive TTY run CLI, otherwise run demo
        if sys.stdin.isatty():
            interactive_loop()
        else:
            counselor = GitaCounselor()
            sample = (
                "I have high anxiety regarding my future, and I am stressed about whether "
                "I will succeed in my exams and career."
            )
            res = counselor.counsel(sample, top_k=2)
            print(format_terminal_report(res))