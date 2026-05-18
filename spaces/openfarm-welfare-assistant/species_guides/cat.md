# Feline Facial Expressions & Visual Well-Being Signals: A Manual for Automated Affective State Assessment

This document serves as a structured cheatsheet and comprehensive guide for assessing feline welfare via visual and behavioral signals. Moving beyond simple pain detection, it is designed for human experts, veterinary practitioners, and developers building Automated Pain Recognition (APR) systems or Large Language Models (LLMs) to perform holistic Affective State assessment in felines.

---

## 1. Holistic Affective State Assessment in Felines

To accurately assess a cat's well-being, observers (and AI models) must distinguish between four primary affective states. Physical pain, emotional distress, and positive welfare often share overlapping visual features that require multi-modal and contextual differentiation.

### 1.1 Negative_Nociceptive (Pain)
Acute or chronic pain caused by tissue damage, surgery, or illness.
*   **Visual Cues (Facial):** Ear flattening/rotation, orbital tightening (squinting), muzzle tension, and forward-spiked whiskers caused by the contraction of the nasolabial levator muscle. [@mota-rojas-2025; q="cats and mice in pain show whisker changes due to the contraction of the nasolabial levator muscle (levator nasolabialis)"]
*   **Visual Cues (Postural/Behavioral):** Hunched posture, tucking of limbs/tail under the body, narrowed and dull eyes, and shifting/restlessness during sleep. [@rojsiripornchai-2025; q="The six Likert-scale items evaluated signs such as: (1) hiding or decreased interest in play or social interaction; (2) poor coat condition or reduced grooming; (3) narrowed and dull eyes; (4) hunched posture or tucking of the tail or limbs under the body; (5) frequent shifting or restlessness during sleep; and (6) increased aggression."]
*   **SoTA Methods:** Feline Grimace Scale (FGS), Glasgow Composite Measure Pain Scale-Feline (CMPS-Feline), UNESP-Botucatu (UFEPS), and the Owner-Assessed Feline Acute Pain Scale. 

### 1.2 Negative_Affective (Stress/Fear)
Psychological distress, anxiety, fear, or restraint stress. 
*   **Visual Cues:** Flattened ears (mimicking pain), dilated pupils (altering orbital shape visibility), lip-licking, and hiding.
*   **Differentiating from Pain:** Fear and stress often lack the specific muzzle tension and whisker spiking seen in nociceptive pain. However, behavioral signs like hiding and aggression overlap heavily with pain. [@rojsiripornchai-2025; q="several items in the scale, such as hiding, decreased social interaction, and increased aggression, may also indirectly reflect the emotional distress associated with pain."]

### 1.3 Positive_Welfare (Joy/Enrichment)
States of comfort, safety, playfulness, and social affiliation.
*   **Visual Cues:** Ears facing forward and upright, round and fully open eyes, relaxed and round muzzle, whiskers curving loosely downward.
*   **Behavioral Cues:** Active grooming (self or allogrooming), playing, resting exposed (not hidden or hunched), and normal eating/drinking behaviors.

### 1.4 Neutral_Resting (Baseline)
Normal physiological resting states, including sleep.
*   **Visual Cues:** Relaxed facial musculature, eyes closed naturally (not forcefully squinted).
*   **Important Constraint:** Facial expressions of pain *cannot* be evaluated during sleep or active grooming. Cats must be awake and undisturbed for accurate baseline or pain assessment.

---

## 2. The Feline Grimace Scale (FGS) - Standard Method

The FGS is the most robust and widely validated facial-expression tool for acute feline pain. It evaluates five specific Facial Action Units (FAUs). 

### Scoring System
Each FAU is scored on a 3-point scale:
*   **0 (Absent):** The pain-related facial action unit is not present.
*   **1 (Moderate/Uncertain):** The AU is moderately present, or there is uncertainty about its presence.
*   **2 (Obvious):** The AU is clearly and obviously present.
*   **Not Possible to Score (N/A):** The AU is obscured or not clearly visible.

**Final Score Calculation:** Sum of assigned scores divided by the maximum possible score (excluding N/A units). 
*Example: If 4 AUs are visible and score a total of 4, the score is 4/8 = 0.5.*

### The 5 Action Units (AUs) Defined
[@evangelista-2019; q="Five AU (ear position, orbital tightening, muzzle tension, whiskers position and head position) were identified."]

1.  **Ear Position:**
    *   *Normal (0):* Ears facing forward.
    *   *Pain (1-2):* Ears are pulled apart, flattened, and rotated outwards/downwards.
2.  **Orbital Tightening:**
    *   *Normal (0):* Eyes fully open and round.
    *   *Pain (1-2):* Narrowing of the orbital area (height between eyelids <50% of eye width), squinting, or tightly closed eyelids.
3.  **Muzzle Tension:**
    *   *Normal (0):* Muzzle is round and relaxed.
    *   *Pain (1-2):* Flattening and stretching of the muzzle from a round to an elliptical shape; the muzzle may appear bulged or tense.
4.  **Whiskers Position:**
    *   *Normal (0):* Whiskers curve loosely downward.
    *   *Pain (1-2):* Whiskers move forward (rostrally) and away from the face, appearing stiff, straight, or "standing on end" (spiked).
5.  **Head Position:**
    *   *Normal (0):* Head is held above or aligned with the shoulder line.
    *   *Pain (1-2):* Head is held below the shoulder line or tilted downward (chin toward the chest).

### Clinical Threshold for Analgesia
A total FGS ratio score of **> 0.39 / 1.0** (approximately 4 out of 10) is the validated cut-off indicating that rescue analgesia (pain relief) is required. [@evangelista-2019; q="The cut-off score of 0.39 (from a maximum of 1.0) was selected"]