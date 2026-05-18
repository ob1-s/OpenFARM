# Canine (Dog) Visual Pain Signals and Facial Expressions Manual
**A Comprehensive Guide for Human Annotators and Automated Systems**

## 1. Introduction and State-of-the-Art (SoTA)
Unlike other species (e.g., cats, mice, horses), **there is currently no single, universally validated "Canine Grimace Scale"** specifically dedicated to facial expressions of pain in dogs [@mota-rojas-2021; q="Although there is no developed and validated grimace scale for dogs"]. 

Instead, the **State-of-the-Art (SoTA) non-automated methods** for assessing discomfort and pain in canines rely on **composite pain scales** that evaluate facial expressions alongside body posture, behavior, and physiological responses. The most prominent tools used by veterinarians include:
*   **CSU-CAPS** (Colorado State University Canine Acute Pain Scale)
*   **CMPS-SF** (Short Form of the Glasgow Composite Pain Scale)
*   **UMPS** (University of Melbourne Pain Scale)
*   **DIVAS** (Dynamic Interactive Visual Analogue Scale)

For automated systems and fine-grained behavioral analysis, the **Dog Facial Action Coding System (DogFACS)** is the gold standard, mapping specific facial muscle movements to Action Units (AUs) and Action Descriptors (ADs).

---

## 2. Visual Indicators of the Four Affective States
Pain and emotional states in dogs trigger neuroendocrine responses (involving the amygdala and motor cortex) that involuntarily contract facial muscles. Identifying the correct affective state requires parsing facial tension, ear positioning, and full-body posture.

### 2.1 Negative_Nociceptive (Pain: Acute/Chronic, Illness)
Physical pain is strongly associated with distinct facial tension and guarded body postures.
*   **Facial Expressions**: Canine pain faces frequently present with "tension of the orbital muscles, low head position, the tension of the facial muscles, retraction of the corner of the lip, and flattening of the ears can be observed" [@mota-rojas-2025; q="tension of the orbital muscles, low head position, the tension of the facial muscles, retraction of the corner of the lip, and flattening of the ears can be observed."].
*   **Body Posture (CSU-CAPS Guidelines)**:
    *   **Score 1 (Mild)**: Content to slightly unsettled. Reacts to palpation by *looking around, flinching, or whimpering*.
    *   **Score 2 (Mild to Mod)**: Looks uncomfortable, droopy ears, worried face. *Flinches, whimpers, cries, or guards/pulls away* from touch.
    *   **Score 3 (Moderate)**: Unsettled, crying, groaning. *Guards or protects wound*, limping, shifting weight. Unwilling to move. 
    *   **Score 4 (Severe)**: Constantly groaning/screaming. *Potentially unresponsive to surroundings*. Rigid body tension. *May bite or chew at wound* [@hellyer-2006; q="Constantly groaning or screaming when unattended"].

### 2.2 Negative_Affective (Stress/Fear)
Stress and fear overlap heavily with pain but are generally driven by environmental triggers rather than nociception.
*   **Visual Cues**: Flattening of the ears or ears rotated backwards/caudally against the head is a reliable indicator of fear and anxiety [@mota-rojas-2021; q="flattening of the ears (EAD105), gestures that indicate warning of an imminent threat"].
*   **Eyes**: Rapid, worried eye movements, or "darting eyes" [@hellyer-2006; q="worried facial expression (arched eye brows, darting eyes)"].
*   **Displacement Behaviors**: Increased blink rate, yawning, and **Lip-Licking / Nose-Licking (AD137)** are frequently associated with appeasement, frustration, or fear [@boneh-shitrit-2022; q="blink, lips part, jaw drop, nose lick, and ears flattener were more common in the negative condition"].
*   **Panting (AD126)**: Rapid respiration through an open mouth, a strong indicator of acute stress or fear when not heat-related.

### 2.3 Positive_Welfare (Joy/Enrichment/Anticipation)
Positive states, such as the anticipation of a food reward or play, are characterized by relaxed, forward-focused attention.
*   **Visual Cues**: Ears are frequently held erect or directed forward (Ears adductor) during positive anticipation.
*   **Facial Relaxation**: An open, relaxed mouth without labial tension, and eyes lacking orbital tightening.

### 2.4 Neutral_Resting (Baseline)
*   **Visual Cues**: Comfortable when resting, happy, content, minimal body tension. Muscles of the face are relaxed without the furrowed brow or lip retraction seen in pain.

---

## 3. Trick Situations, Edge-Cases, and Confounders (CRITICAL)
Automated systems and human annotators must rigorously account for morphological diversity, physiological states, and contextual masking.

### A. Morphological Confounders (Breed Differences)
*   **Ear Shape**: Floppy ears (e.g., Bassets, Retrievers) naturally obscure ear base movement and are much harder to assess for pain/fear than erect ears [@appels-2025; q="floppy ears are more easily recognised by the model compared to floppy-eared dogs"].
*   **Snout Length & Facial Folds**: Brachycephalic or prognathic dogs have deep permanent facial folds. These wrinkles mimic pain-induced "furrowed brows" or cheek tension, creating false positives [@mota-rojas-2021; q="cranial structural features of prognathic dogs with short muzzles and deep facial folds do not allow adequate recognition"].
*   **Coat Color and Texture**: 
    *   *Dark/Black Fur*: Obscures shadows and micro-expressions, hiding muscle tension in the supra/infraorbital regions.
    *   *Long Fur*: Masks facial landmarks entirely [@boneh-shitrit-2022; q="breeds with long fur covering facial features or extending on ears are the most difficult to detect accurately"].

### B. Physiological Confounders
*   **Sleep Status and Sedation**: Pharmacological agents and normal sleep alter facial muscle tone. Assessing sleeping or heavily sedated dogs generates false positives for grimace scales [@cohen-2020; q="False positives are known to occur in a small range of scenarios such as sedation/anaesthesia, sleeping status"].
*   **Tiredness vs. Pain**: Exhaustion closely mimics pain expressions. In working dogs, "backward ears and tension or stiffness in the muscles around the eyes might be present due to tiredness instead of pain" [@mota-rojas-2025; q="backward ears and tension or stiffness in the muscles around the eyes might be present due to tiredness instead of pain"].
*   **Life-Stage (Neonates)**: Facial expressions may not map 1:1 between adults and puppies. For automated systems, "Future studies should consider if neonates have the same changes and FAUs." [@mota-rojas-2025; q="Future studies should consider if neonates have the same changes and FAUs."]

### C. Contextual/Environmental Confounders
*   **Chronic Pain Adaptation**: Facial scales are excellent for acute pain, but over time, animals adapt and mask expressions. "Animals suffering from chronic pain are unlikely to maintain a certain expression in the long term" [@mclennan-2019; q="Animals suffering from chronic pain are unlikely to maintain a certain expression in the long term"]. This results in a **False Negative** for nociceptive pain.
*   **The Observer Effect & Paedomorphic Traits**: Animals frequently alter their expression based on who is watching. Prey species often hide pain [@mclennan-2019; q="prey species do not overtly express signs of pain and distress when potential predators such as humans are present"]. Conversely, dogs may actively contract the *levator anguli oculi medialis* (AU101) to solicit human attention. Raising the inner brow enlarges the eyes, causing humans to falsely interpret the expression as sadness or pain [@mota-rojas-2021; q="AU101 muscle... enhances visual contact and confers a child-like appearance"].
