# 🐰 Rabbit Pain Assessment & Grimace Scale Manual
**Target Class:** `Rabbit` (*Oryctolagus cuniculus*)
**Primary Frameworks:** Rabbit Grimace Scale (RbtGS), Composite Pain Scale for Rabbits (CANCRS)

This manual provides structured instructions, definitions, and edge-cases for holistically recognizing the affective states of rabbits. It is designed as a foundational cheat-sheet for both human veterinary practitioners and the development of Automated Pain Recognition (APR) systems using Computer Vision (CV) and Large Language Models (LLMs).

---

## 1. The Four Affective States

### 1.1 Negative_Nociceptive (Pain)
Acute somatic or surgical pain triggers clear, measurable shifts in facial musculature and body posture. 
*   **Facial Signs:** Characterized by orbital tightening, cheek flattening, a "V" shaped pointed nose, and stiff, horizontal or backward-facing whiskers. [@keating-2012; q="rabbits display a cheek and nose bulge at rest, which become progressively pointed or flattened as the intensity of pain increases"]
*   **Behavioral Postures:** Noxious stimuli often elicit specific nocifensive behaviors. Key postures include **Pressing** ([@keating-2012; q="Abdomen pushed towards floor, usually before walking"]), **Arching** ([@keating-2012; q="Full arching of the back upwards"]), and **Wincing** ([@keating-2012; q="Rapid movement of the backwards in a rocking motion accompanied by eye closing and swallowing action"]). 
*   **Assessment Tools:** Scored via the RbtGS (facial) and CANCRS (composite).

### 1.2 Negative_Affective (Stress/Fear)
Because rabbits are a prey species, their primary defense mechanism against stress, anxiety, or potential predation is "conservation withdrawal". 
*   **Visual Cues:** The rabbit becomes completely silent and immobile ("freezing"), actively masking overt signs of distress to avoid drawing attention. [@shaw-2020; q="rabbits, as a prey species, can mask symptoms as a survival mechanism to avoid appearing weak"]
*   **Differentiation from Pain:** While pain often presents with specific postural contortions (arching, pressing, writhing) and distinct facial grimacing, extreme fear presents as rigid, tense immobility often with wide, alert eyes (hypervigilance) rather than the orbital tightening seen in pain.

### 1.3 Positive_Welfare (Joy/Enrichment)
Positive welfare goes beyond the mere absence of pain. 
*   **Visual Cues:** The rabbit is highly active, engaging in **Exploration** ([@hampshire-2015; q="Playful, curious, ambulatory"]). Normal grooming behaviors are frequently observed, and the animal eagerly accepts food/treats. 
*   **Facial Signs:** Cheeks are fully puffed/rounded, eyes are wide open, and ears are held alert and upright (or in their breed-specific relaxed position).

### 1.4 Neutral_Resting (Baseline)
The baseline physiological state where the animal is at rest but not distressed. 
*   **Visual Cues:** The rabbit may be lying down with legs tucked under the body or sprawled on its side. 
*   **Facial Signs:** Facial muscles are relaxed. The nares form a soft "U" shape, and whiskers exhibit a gentle, natural downward curve. 

---

## 2. Rabbit Grimace Scale (RbtGS) - Facial Action Units (FAUs)
The RbtGS evaluates five Facial Action Units (FAUs). Each FAU is scored on a 3-point scale (0 = Normal, 1 = Moderate, 2 = Severe).

### A. Orbital Tightening (Eye Narrowing)
*   **Score 0:** Eyes are wide open and alert.
*   **Score 1:** Eyes are slightly narrowed. The eyelid is partially closed.
*   **Score 2:** Eyes are overtly tightened or completely closed. The globes may be drawn inward. 

### B. Cheek Flattening
*   **Score 0:** Cheeks are puffy and rounded.
*   **Score 1:** Cheeks are slightly flattened; there is less definition between the cheeks and the nose.
*   **Score 2:** Whiskers and cheeks are markedly flat. The rounded appearance of the cheeks is lost.

### C. Nose Shape
*   **Score 0:** The nose appears relaxed, and the nares form a rounded "U" shape.
*   **Score 1:** The nares are slightly drawn, beginning to form a "V" shape.
*   **Score 2:** The nares are drawn vertically, creating a sharply pointed nose that resembles a distinct "V". The nose is drawn down toward the chin.

### D. Whisker Position
*   **Score 0:** Whiskers have a gentle, natural downward curve.
*   **Score 1:** Whiskers are flattened against the cheeks or oriented caudally (backward).
*   **Score 2:** Whiskers are straightened and extended horizontally, or pulled back tightly against the face.

### E. Ear Position & Shape
*   **Score 0:** Ears are alert, upright, or pointing in a specific direction of attention.
*   **Score 1:** Ears are droopy in the frontal view and oriented caudally at a ~60º angle in the profile view. [@hampshire-2015; q="Ears are droopy in frontal view and oriented caudally at a 60º angle in profile view"]
*   **Score 2:** Ears are flattened in both frontal and profile views, rotated to face towards the hindquarters, and tightly folded or curled. Head is often tucked against the chest.

---

## 3. Pain Score Thresholds (CANCRS & RbtGS)
To standardize intervention, pain is grouped into four clinical classes based on the summation of scores. 

**Standard RbtGS Thresholds (Max Score = 10):**
*   **0 – 1:** No Pain (NP)
*   **2 – 4:** Discomfort (D)
*   **5 – 7:** Moderate Pain (MP) - *Analgesic intervention heavily recommended*
*   **8 – 10:** Severe Pain (SP)

**CANCRS Composite Thresholds (Max Score = 24):**
*   **0 – 5:** No Pain (NP)
*   **6 – 11:** Discomfort (D)
*   **12 – 17:** Moderate Pain (MP)
*   **18 – 24:** Severe Pain (SP)

---

## 4. Trick Situations, Edge Cases & Algorithmic Confounders

### A. Morphological Variations (Lop-Eared Breeds)
*   **The Issue:** Lop-eared rabbits have ears that naturally fall into a "droopy" or flattened position.
*   **Solution:** The Ear Position FAU **must be excluded** when assessing lop-eared rabbits. For automated systems, breed-detection classifiers must first route lop-eared rabbits to an adjusted RbtGS scoring out of a maximum of 8 points rather than 10. [@banchi-2020; q="lop eared rabbits’ ear morphology does not permit for evaluation of ear position properly"]

### B. Visceral Pain vs. Somatic Pain (The "Ileus" Edge Case)
*   **The Issue:** RbtGS is highly validated for acute somatic/surgical pain (e.g., ear tattooing, castration, orthopedic surgery). However, clinical studies show it frequently fails to detect severe visceral pain, such as Gastrointestinal Stasis (Ileus). [@shaw-2020; q="rabbit 11 had been diagnosed with ileus"]
*   **Solution:** In cases of suspected visceral pain, RbtGS alone is insufficient. Systems and humans must rely on holistic parameters (e.g., hunched posture, teeth grinding/bruxism, abdominal pressing).

### C. The Chronicity Mask (Adaptation)
*   **The Issue:** Over time, animals suffering from prolonged/chronic pain often stop displaying the overt acute facial expressions measured by grimace scales.
*   **Solution:** A neutral facial expression in a rabbit with a chronic injury is a **False Negative**. Facial scales are most reliable for *acute* pain. [@mclennan-2019; q="Animals suffering from chronic pain are unlikely to maintain a certain expression in the long term"]

### D. Environmental & Visual Confounders
*   **Coat Color / Contrast:** Assessing whisker position is notoriously difficult in albino/white rabbits (e.g., New Zealand White) when recorded against a white background or in brightly lit clinical environments. CV systems must apply high-contrast filters or edge-detection specifically tuned for whiskers. [@shaw-2020; q="white colouration against a white background, which made it difficult to visualise the whisker position"]
*   **Observer Effect (Freezing):** Rabbits will actively mask pain and "freeze" when a human approaches the cage. 
*   **Solution:** Live, cage-side scoring is heavily prone to false negatives. Video-based or remote sensor scoring (allowing the rabbit to believe it is unobserved) is the gold standard for accurate assessment. [@mclennan-2019; q="many prey species do not overtly express signs of pain and distress when potential predators such as humans are present"]