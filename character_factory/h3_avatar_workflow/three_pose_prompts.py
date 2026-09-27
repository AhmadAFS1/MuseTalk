"""Identity-adapted H3 prompts with the selected pose motion unchanged."""
from create_avatar import prompt as accepted_talking_prompt

VERSION = 'h3-three-pose-20260927-v1'
POSES = ('idle', 'talking', 'smiling')
NATIVE_FRAMES = {'idle': 243, 'talking': 243, 'smiling': 158}
DELIVERY_FRAMES = {'idle': 240, 'talking': 240, 'smiling': 158}


def identity(profile: dict) -> dict:
    pronoun = profile.get('pronouns', {'woman': 'she', 'man': 'he'}.get(profile.get('gender'), 'they')).lower()
    if pronoun not in ('she', 'he', 'they'):
        raise ValueError('pronouns must be she, he, or they')
    possessive = {'she': 'her', 'he': 'his', 'they': 'their'}[pronoun]
    subject = profile.get('subject')
    if not subject:
        if all(k in profile for k in ('label', 'age', 'hair', 'tone')):
            subject = f"a {profile['label']}, age {profile['age']}, with {profile['hair']} and {profile['tone']}"
        else:
            subject = 'an adult'
    return dict(subject=subject, wardrobe=profile.get('wardrobe', 'the exact clothing in the reference image'),
                pronoun=pronoun, possessive=possessive,
                facial_hair=profile.get('facial_hair', 'none'))


def pose_prompt(profile: dict, pose: str) -> str:
    if pose not in POSES:
        raise ValueError(f'Unknown pose: {pose}')
    if pose == 'talking' and all(k in profile for k in ('label', 'age', 'gender', 'hair', 'tone', 'facial_hair')):
        # Preserve the exact accepted prompt builder for fully described subjects.
        return accepted_talking_prompt(profile)
    a = identity(profile)
    pro, poss = a['pronoun'], a['possessive']
    Pro, Poss = pro.capitalize(), poss.capitalize()
    is_plural = pro == 'they'
    if pose == 'talking':
        identity_line = (f"Vertical 9:16 photorealistic FaceTime close-up. The same {a['subject']} from the reference image, "
                         f"wearing {a['wardrobe']}, in the same room. {Pro} {'look' if is_plural else 'looks'} straight into the lens "
                         f"and {'speak' if is_plural else 'speaks'} this exact line in a calm friendly voice: "
                         '"Hi, I\'m glad you\'re here. Let me show you how this works." ')
        motion = ('Natural blinking, a nearly still head held at the reference angle throughout, with only tiny natural micro-movements. '
                  'Keep the head centered at a constant size, shoulders level, and gaze into the lens. Keep head pitch, yaw, and roll nearly constant: '
                  'no nodding, chin lifts, head tilts, leaning, or swaying. The camera and framing remain fixed. '
                  'Maintain natural speaking lip and jaw articulation, coherent teeth and lips, stable identity, unchanged clothes and room, no captions, no logos. '
                  f'The person in <Picture 1> is the only person on camera. Keep {poss} face, hair, and clothing consistent.')
        prompt = identity_line + motion
    else:
        # Motion sentences are adapted only for pronouns from the saved FL2VA
        # idle/smile prompts, not from the older LTX three-pose prompt pack.
        opening = (f"Vertical 9:16 photorealistic UGC smartphone video. The exact same person from the reference image, "
                   f"{a['subject']} wearing {a['wardrobe']}, framed in a tight medium close-up from the chest up at arm's length, "
                   f"facing the camera on a video call. The camera is locked: no zoom, no reframing, no change in subject size. "
                   f"{Poss} mouth stays closed and relaxed throughout. Natural skin texture, realistic eyes and hair, stable identity, "
                   'fixed background and lighting. The person in the supplied keyframe image is the only person on camera. '
                   f"Keep {poss} face, hair, clothing, room, and lighting consistent with the supplied keyframe image. ")
        if pose == 'idle':
            motion = (f"{Pro} {'rest' if is_plural else 'rests'} in a settled upright posture and {'look' if is_plural else 'looks'} into the lens with quiet friendly attention. "
                      f"{Poss} chest rises and falls slowly with breathing. One blink occurs, unhurried. "
                      f"{Poss} attention settles slightly forward and {poss} head develops one gradual diagonal listening inclination, "
                      f"held quietly while {pro} {'keep' if is_plural else 'keeps'} eye contact. A single small acknowledgement occurs inside that already-established posture. "
                      f"Breathing continues slowly. The diagonal softens and {poss} head returns along the same path toward the centre. "
                      f"{Pro} {'settle' if is_plural else 'settles'} back into the exact resting posture the clip opened with, breathing slowly, mouth closed, eyes on the lens.")
        else:
            motion = (f"{Pro} {'rest' if is_plural else 'rests'} with a neutral relaxed face looking into the lens, breathing slowly. "
                      f"A mild closed-lip smile grows, accompanied by a tiny lift of {poss} chin, and holds briefly. "
                      f"{Poss} lips stay together and {poss} eyes stay warm on the lens. The smile and the chin lift release fully and "
                      f"{pro} {'return' if is_plural else 'returns'} to the neutral resting face and posture the clip opened with, eyes on the lens, breathing slowly.")
        prompt = opening + motion
    if a['facial_hair'] not in ('none', 'clean-shaven'):
        prompt += f" Maintain {poss} {a['facial_hair']} consistently, including individual hairs and the visible lip border."
    return prompt
