CHEATSHEET = """
# Sheep & Lamb Facial Expressions / Grimace Pain Scale Manual
**For Automated Pain Recognition (APR) Systems**

This document provides structured instructions, precise definitions, and edge-case parameters for training Large Language Models (LLMs) and Computer Vision (CV) systems to recognize acute pain in ovine species (sheep and lambs) based on facial action units (FAUs) and grimace scales.

---

## 1. Standard Recognized Grimace Scales in Ovine Species
Pain in sheep is assessed through three primarily recognized scales, differentiated by age (lambs vs. adult sheep) and specific pain models. 

### A. Sheep Grimace Scale (SGS) 
*Source: Häger et al., 2017 ("The Sheep Grimace Scale as an indicator of post-operative distress and pain in laboratory sheep")*
Developed using adult sheep undergoing tibia osteotomy. It relies on 3 FAUs.
*   **Orbital tightening:** 
    *   Score 0 (Not present): "widely opened eyes"
    *   Score 1 (Moderate): "half-closed eyes were defined as an expression of “moderate” pain"
    *   Score 2 (Severe): "completely closed eyes were assigned with “severe” pain"
*   **Ear and head position:** 
    *   Score 0 (Not present): "straightened ears and heads"
    *   Score 1 (Moderate): "flattened ears and a slanted head were defined as an expression of moderate pain"
    *   Score 2 (Severe): "hanging ears and head were defined as severe"
*   **Flehming / Lip puckering:** *(Note: Unique scoring system due to head movement)*
    *   Score 0 (Not present): "closed mouth"
    *   Score 1 (Moderate): "puckered lips were defined as an expression of moderate pain with a score of 1"
    *   Score 3 (Severe): "Flehming represents a sign of severe pain... Due to the fact that sheep lift up their head during flehming, the position of the head cannot be used for evaluation at this time-point and flehmen was therefore matched with a score of 3."

### B. Sheep Pain Facial Expression Scale (SPFES)
*Source: McLennan et al., 2016 (referenced in Chiavaccini et al., 2024 and McLennan et al., 2019)*
Developed for adult sheep using naturally occurring diseases (footrot and mastitis). It utilizes 6 FAUs:
1. Orbital region opening
2. Cheek tightening (masseter muscle prominence)
3. Ear frontal position
4. Ear side position
5. Lip and jaw profile
6. Nostril and philtrum shape

### C. Lamb Grimace Scale (LGS)
*Source: Guesgen et al., 2016 ("Coding and quantification of a facial expression for pain in lambs")*
Developed using 5-to-6-week-old lambs undergoing routine tail-docking. It utilizes 5 FAUs scored on a 0-2 scale (0 = not present, 1 = moderately present, 2 = obviously present).
*   **Orbital Tightening:** "Lambs in pain show “squeezing” of the eye or closing of the eye... If the eye closure reduces the visibility of the eye by more than half, it would be scored as obvious (2)."
*   **Nose Features:** "The nose of lambs in pain appears tightened with a decrease in nostril size... Flattening makes the nose appear more like a horizontal line in frontal headshots, whereas pointing makes the nose appear more ‘V’ rather than ‘U’ shaped in frontal headshots."
*   **Mouth Features:** "The lips of a lamb in pain are flattened and tightened. The lips appear more like a horizontal line in frontal headshots. There is lack of the ‘upwards curl’ at the edge of the lips that gives lambs their ‘smiling’ appearance when not in pain."
*   **Cheek Flattening:** "Lambs in pain show less bulging of the nose and cheek area. In obvious cases, the cheek has a ‘hollowed’ appearance. When not in pain, the cheeks appear rounded"
*   **Ear Posture:** "Lambs in pain have ears that are tense and point backwards or downwards so that the inner part of the ear is not visible. As a result, ears may appear narrower, and flattened dorsally. When lambs are not in pain, their ears are relaxed and horizontal, or slightly forward of the head and the inner ear is visible."

---

## 2. Divergence Across Life Stages (Lambs vs. Adult Sheep)
Automated systems must be programmed to conditionally evaluate specific FAUs based on the age of the subject, as expressions diverge ontogenetically.

*   **Ear Posture Divergence:**
    *   *Lambs:* "The ears of lambs point backwards when in pain" (McLennan et al., 2019, "Conceptual and methodological issues relating to pain assessment in mammals").
    *   *Adult Sheep:* "whereas in adult sheep the ears rotate ventrally and caudally" (McLennan et al., 2019).
*   **Cheek/Masseter Divergence:**
    *   *Lambs:* The cheek area exhibits flattening or a "hollowed appearance" (Guesgen et al., 2016).
    *   *Adult Sheep:* The masseter muscle becomes prominent and bulging; "whereas in adult sheep the masseter muscle becomes more prominent." (McLennan et al., 2019).

---

## 3. Trick Situations, Edge Cases, and Confounders

### A. Environmental Restraint vs. Free Roaming
Physical restraint modifies facial action units, particularly ears, overriding pain signals due to the animal's need to gather environmental information.
*   **Edge Case Definition:** "Freely moving lambs were scored lower using the LGS over both periods and had a significantly smaller eye aperture and smaller nose and ear angles than when they were held." (Guesgen et al., 2016).
*   **False Negatives in Ears:** "Lambs in the pen may have had their ears forward to gather information about their conspecifics... this FAU may have had the strongest effect on LGS." (Guesgen et al., 2016). *System Rule: If the sheep is in a pen interacting with humans/conspecifics, ear position reliability decreases.*

### B. The "Flehming" Override (Adult Sheep)
Flehming (curling of the upper lip exposing teeth) is a behavioral response usually associated with olfactory investigation, but in severe pain, it serves as a critical pain indicator.
*   **Edge Case Definition:** During Flehming, the animal tilts its head upwards, obscuring the standard baseline head posture.
*   **System Rule:** If Flehming is detected, the system must automatically assign maximum pain severity (Score 3 in SGS) and *suspend* the requirement to evaluate "head position" which "cannot be used for evaluation at this time-point" (Häger et al., 2017).

### C. Analgesia, Sedation, and Anesthesia
Anesthetics and sedatives relax facial muscles, artificially altering pain scores. 
*   **Edge Case Definition:** "Behavior changes caused by sedation and pain may be similar, therefore sedation may be a confounding factor when assessing pain in all species and may increase the chance of a false positive pain diagnosis." (da Silva et al., 2025, "Clinical validation of the Unesp-Botucatu acute pain scale").
*   *System Rule:* Pain scoring must account for the half-life of sedative agents; do not score immediately post-anesthesia unless specifically validating anesthetic recovery protocols.

### D. Asymmetry and Camera Angle
Sheep facial expressions can be highly lateralized (asymmetrical). 
*   **Edge Case Definition:** "Orbital tightening... may only occur, or occur more strongly, in one eye." (Guesgen et al., 2016).
*   *System Rule:* Computer Vision pipelines must evaluate both sides of the face independently. If a single camera is used, front-on angles are mandatory.

### E. Breed and Phenotypic Variations
*   **Edge Case Definition:** Long wool, dark coats, and differing facial anatomies cause occlusions.
*   "Observers had more difficulty assigning scores to long-haired ferrets... the current authors have all experienced difficulties in assessing the facial expression of longer-haired, muscular, darker-haired animals... difficult to determine if certain features are changing due to shadow or some other element." (McLennan et al., 2019).
*   *System Rule:* The model must adjust confidence thresholds when assessing dark-faced breeds (e.g., Suffolk, Blackface) or heavily wooled faces, relying more heavily on geometric angles (e.g., ear base angle, eye aperture) rather than texture/shadow-based feature extraction.

"""
