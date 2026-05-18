# Equine Visual Pain Recognition & Grimace Scale Manual
**Target Audience**: Equine Veterinarians, Clinical Researchers, and Automated Pain Recognition (LLM/CV) Systems.
**Species Focus**: Equine (*Equus caballus*), including adult horses and foals. (Note: Donkeys require separate ethograms due to behavioral divergence).

---

## 1. Introduction & State-of-the-Art (SoTA) Methods

Equine pain assessment relies heavily on recognizing subtle behavioral and facial changes, as horses are stoic prey animals that naturally conceal vulnerabilities. The State-of-the-Art (SoTA) non-automated methods for assessing discomfort rely on structured composite and facial grimace scales. 

The primary validated systems for facial pain and behavioral assessment are:
1. **Horse Grimace Scale (HGS)**: Developed for post-surgical/acute pain (e.g., castration, laminitis, dental pain). [@dalla-costa-2014; q="The Horse Grimace Scale (HGS) was developed, and comprises six facial action units"]
2. **Equine Pain Face**: Identifies facial characteristics of induced acute pain. [@gleerup-2015; q="An equine pain face comprising ‘low’ and/or ‘asymmetrical’ ears, an angled appearance of the eyes"]
3. **Equine Utrecht University Scale for Facial Assessment of Pain (EQUUS-FAP)**: Designed for visceral (colic), head-related, and orthopedic pain. [@van-loon-2015; q="The EQUUS-FAP is a multifactorial SDS based on nine parameters"]
4. **EquiFACS (Equine Facial Action Coding System)**: The anatomical, objective gold-standard coding system from which clinical grimace scales are derived. [@andersen-2021; q="The FACS standard for horses, EquiFACS, was developed in 2015"]
5. **Equine Acute Abdominal Pain Scale (EAAPS) & Composite Pain Scale (CPS)**: Composite scales combining behavioral postures (e.g., flank watching, weight distribution) with facial responses for visceral and orthopedic pain, respectively. [@sutton-2013; q="Flehman (lip curling), rolling, pawing, sternal recumbency, lateral recumbency, attempting to lie down, stretching, flank watching and kicking abdomen"]

---

## 2. The Standard Horse Grimace Scale (HGS) & EquiFACS Variations

The HGS is the most widely adopted framework. It assesses six Facial Action Units (FAUs). 
**Scoring System**: 
* **0**: Not present (High confidence AU is absent)
* **1**: Moderately present (Moderate appearance or equivocation)
* **2**: Obviously present (Marked appearance)

### Core Facial Action Units (FAUs)
1. **Stiffly Backwards Ears vs. Ear Adduction**: HGS looks for ears held stiffly and rotated backward. However, EquiFACS analysis reveals that "lowered ears" are often a sign of rest, whereas actual pain is better predicted by ear *adduction* (ears pulled toward the midline) and asymmetrical rotation. [@ask-2024; q="flattening and rotating both ears (EAD103 and EAD104) instead were associated with no pain, which is in agreement with previous findings that lowered ears can be seen during sleep"]
2. **Orbital Tightening**: The eyelid is partially or completely closed, reducing the eye size by more than half. 
   * *Methodological Difference*: While HGS focuses on orbital tightening, the **Equine Pain Face** model notes an "angled eye" with a tense stare, where tension in the *m. levator anguli oculi medialis* gives the upper eyelid an angled appearance, sometimes exposing the sclera. [@gleerup-2015; q="giving the upper eyelid a very characteristic angled appearance which increased the incidence of exposing the sclera"]
3. **Tension Above the Eye Area**: Contraction of the muscles above the eye causes increased visibility of the underlying bone surfaces (temporal crest bone is clearly visible).
4. **Prominent Strained Chewing Muscles**: Straining chewing muscles are clearly visible as an increased tension above the mouth. 
5. **Mouth Strained and Pronounced Chin**: The upper lip is drawn back, and the lower lip causes a pronounced, flat chin. 
   * *EquiFACS Extension*: In orthopedic models, precise mouth movements like *jaw thrust* and *lips part* strongly co-occur with pain. [@ask-2024; q="lower lip depressor (AU16), lips part (AU25), half blink (AU47), single ear forward (SEAD101) and single ear rotator (SEAD104) were selected as co-occurring significantly more in horses with pain"]
6. **Strained Nostrils and Flattening of the Profile**: Nostrils look strained and slightly dilated mediolaterally. The profile of the nose flattens, and lips elongate. [@dalla-costa-2014; q="strained nostrils and flattening of the profile"]

---

## 3. Additional Facial & Behavioral Posture Indicators 

Beyond the face, whole-body behaviors differentiate the specific source of negative nociceptive or affective states:

### A. Visceral Pain (Colic) Postures
* **Flank Watching / Kicking Abdomen**: The horse repeatedly glances at its flank or kicks in the direction of its abdomen. [@sutton-2013; q="Flehman (lip curling), rolling, pawing, sternal recumbency, lateral recumbency, attempting to lie down, stretching, flank watching and kicking abdomen"]
* **Stretching / Rolling**: Frequent stretching (similar to a urination stance without urinating) or attempting to roll on the ground.

### B. Orthopedic Pain Postures
* **Weight Shifting / Pointing**: Frequently shifting weight off a specific limb, or extending one forelimb cranially to reduce weight-bearing ("pointing"). [@bussieres-2008; q="altered weight distribution and altered limb loading/position"]
* **Atypical Recumbency**: Lying in full lateral recumbency without sleeping, often accompanied by groaning.

### C. Additional Facial Indicators
* **Focus/Demeanor**: Less focused on the environment, displaying a "zoned out," worried, or glassy-eyed stare. [@torcivia-2021; q="Less responsive to the environment, often with “zoned out,” worried, or glassy-eyed staring"]
* **Flehmen Response**: Lifting the head with curling of the upper lip. Spontaneous flehmen out of context is a strong pain indicator in adults. [@torcivia-2021; q="with discomfort, flehmen may occur out of that context."]
* **Yawning / Teeth Grinding / Moaning**: Repeated yawning bouts, bruxism, or groaning.

---

## 4. Life Stages: Foals vs. Adult Horses

Neonatal and older foals (0-6 months) require an adapted scale (**EQUUS-FAP FOAL**) because their normal baseline behaviors differ significantly from adults. [@lanci-2022; q="physiological and behavioural differences between foals and mature horses; consequently, discomfort ethograms are unlikely to be fully similar"]

**Key Differences for Foals:**
1. **Sclera Visibility**: Removed from the foal pain scale. *Rule*: Many healthy, pain-free foals naturally show their sclera. [@lanci-2022; q="visibility of the sclera was omitted from the facial action unit “eyelids” because many healthy pain-free foals also show their sclera"]
2. **Flehmen Response**: Removed from the foal pain scale. *Rule*: Foals frequently perform the flehmen response during normal, pain-free environmental exploration.
3. **Lip Smacking**: While adult horses show "mouth strained," foals display "licking/chewing" rather than submissive lip smacking when in pain. 
4. **Recumbency**: Foals naturally spend much more time recumbent (lying down) than adults. Recumbency alone is not a pain indicator unless accompanied by generalized muscle tremors or restlessness.

---

## 5. Confounders, Edge-Cases, and "Trick Situations"

Automated systems and human raters must account for high-variance environmental and physiological confounders:

### A. The Observer Effect (Prey Animal Stoicism)
Horses reflexively mask pain in the presence of humans/predators. 
* **Edge-case**: A horse may display severe pain signs on remote video but immediately halt these behaviors and appear "alert" when a human enters the stall. 
* **Protocol**: Pain assessments and image capture *must* be done remotely (via camera) or before interacting with the horse. [@torcivia-2021; q="ongoing discomfort behaviors diminished or stopped altogether during caretaker visits"]

### B. Anesthesia & Drug "Hangovers"
* **Trick Situation**: Inhalant anesthesia and sedatives artificially alter facial muscle tone for hours post-procedure, completely mimicking grimace markers.
* **Edge-case**: Alpha-2 agonists (xylazine) cause droopy ears, relaxed lower lips, and half-closed eyes. This mimics "stiffly backwards ears" and "orbital tightening". [@gleerup-2015; q="resembles the appearance of the horses dozing off in the present study, suggesting that the Horse Grimace Scale describes a combination of pain and fatigue"] Additionally, general anesthesia artifacts can linger. [@chiavaccini-2024; q="facial expression of pain can remain altered for several hours after inhalant anesthesia"]

### C. Fatigue/Sleep vs. Pain
* **Trick Situation**: Normal resting or sleeping behaviors share FAUs with pain.
* **Differentiation**: Lowered, flattened ears without tension are a sign of *sleep*, not pain. EquiFACS specifically links true pain to *ear adduction* and *rotation*, not just "lowered" ears. [@ask-2024; q="flattening and rotating both ears (EAD103 and EAD104) instead were associated with no pain, which is in agreement with previous findings that lowered ears can be seen during sleep"] Also, tiredness causes tension around the eyes. [@mota-rojas-2025; q="appearance of FAU such as backward ears and tension or stiffness in the muscles around the eyes might be present due to tiredness instead of pain"]

### D. Stress/Fear vs. Pain
* **Trick Situation**: A horse experiencing fear will show widened eyes (sclera visible) and flared nostrils, triggering false positives. 
* **Differentiation**: Pain is characterized by orbital *tightening* (squinting) and a *withdrawn* stare, whereas fear features an *alert, hyper-responsive* stare. [@nasem-2021; q="stress-induced analgesia or hypoalgesia and is considered typical in prey animals"]

### E. Chronic Pain Adaptation
* **Trick Situation**: A horse with a known long-term injury or chronic condition displays a completely neutral face (HGS = 0).
* **Explanation**: Chronic pain expressions fade over time as the animal behaviorally adapts. Grimace scales are highly reliable for *acute* pain, but yield False Negatives for *chronic* pain. [@mclennan-2019; q="Animals suffering from chronic pain are unlikely to maintain a certain expression in the long term as pain can fluctuate over time"]

### F. Mild vs. Moderate Pain (Non-Linearity)
* **Edge Case**: A horse transitions from mild to moderate pain, but its facial grimace score *stops increasing* or even drops slightly.
* **Explanation**: The transition from "no pain" to "mild pain" produces the most dramatic facial changes. As pain becomes "moderate," the horse may adopt relieving body postures (shifting weight, recumbency) that actually *reduce* facial tension, meaning facial scales do not linearly scale with pain severity at the higher end. [@ask-2024; q="major change in facial activity occurred in the transition from no pain to mild pain"]

### G. Anatomy and Coat Color
* **Trick Situation**: Dark-coated horses (black, dark bay) are significantly harder to score for subtle FAUs like "orbital tightening" and "prominent strained chewing muscles" due to lack of contrast and shadows. [@dalla-costa-2014; q="horses with dark-brown or black coats were more difficult to score than grey and light brown coat"]
