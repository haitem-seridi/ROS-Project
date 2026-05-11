"""
fsm.py — Machine à États Finis robuste pour robot RÉEL
Gère les transitions entre tous les challenges :
  IDLE → LINE_FOLLOWING → ROUNDABOUT → OBSTACLE_STOP
  + intégration Challenge 2 (avoidance), 3 (corridor), 4 (soccer)

Adapté pour TurtleBot3 réel :
  - Seuils de courbure relevés pour ignorer le bruit caméra
  - Confirmation sur 15+ frames pour le rond-point
  - Logique d'évitement plus robuste (timer minimum + LIDAR dégagé)
"""

from enum import Enum, auto
import time


class State(Enum):
    IDLE              = auto()
    LINE_FOLLOWING    = auto()
    ROUNDABOUT        = auto()
    OBSTACLE_STOP     = auto()
    OBSTACLE_AVOID    = auto()   # Challenge 2 : contournement actif
    CORRIDOR          = auto()   # Challenge 3 : navigation couloir LIDAR
    SOCCER_SEARCH     = auto()   # Challenge 4 : recherche balle
    SOCCER_APPROACH   = auto()   # Challenge 4 : approche balle
    SOCCER_PUSH       = auto()   # Challenge 4 : pousse vers but


class ChallengeFSM:
    """
    FSM centralisée pour tous les challenges.

    Transitions :
    - IDLE       → LINE_FOLLOWING  : appui touche [s]
    - LF         → ROUNDABOUT     : courbure > seuil pendant N frames
    - ROUNDABOUT → LF              : timer expiré
    - LF/RA      → OBSTACLE_STOP  : obstacle < seuil_stop (challenge 1)
    - OBSTACLE_STOP → LF           : obstacle dégagé
    - LF/RA      → OBSTACLE_AVOID : challenge 2 activé + obstacle < seuil_avoid
    - OBSTACLE_AVOID → LF          : timer minimum écoulé ET obstacle dégagé

    Paramètres clés exposés pour tuning live via trackbars :
      CURVATURE_THRESHOLD : coeff a du poly qui déclenche le rond-point
      ROUNDABOUT_DURATION : durée fixe de traversée du rond-point (secondes)
      OBSTACLE_STOP_DIST  : distance LIDAR → arrêt d'urgence (m)
    """

    def __init__(self,
                 roundabout_direction: str = 'right',
                 challenge: int = 1):

        self.state                = State.IDLE
        self.roundabout_direction = roundabout_direction   # 'left' | 'right'
        self.challenge            = challenge              # 1, 2, 3, 4

        # ── Paramètres tunable ────────────────────────────────────────────
        # IMPORTANT : ces valeurs sont calibrées pour un TurtleBot3 RÉEL.
        # En simulation Gazebo, on peut baisser les seuils.
        self.CURVATURE_THRESHOLD  = 0.003   # seuil coeff a pour rond-point
                                             # (ancien: 0.0008, beaucoup trop bas)
        self.ROUNDABOUT_DURATION  = 5.0     # secondes traversée rond-point
        self.OBSTACLE_STOP_DIST   = 0.30    # mètres → arrêt urgence
                                             # (ancien: 0.25, trop juste pour un vrai robot)
        self.OBSTACLE_AVOID_DIST  = 0.60    # mètres → début contournement (C2)
                                             # (ancien: 0.50, pas assez d'anticipation)
        self.CURVATURE_CONFIRM_N  = 15      # N frames consécutives pour confirmer
                                             # (ancien: 5, soit 0.16s, beaucoup trop peu)
        self.AVOID_MIN_DURATION   = 2.0     # secondes minimum en état AVOID
                                             # (empêche de quitter l'évitement trop tôt)
        self.AVOID_MAX_DURATION   = 6.0     # secondes maximum en état AVOID
                                             # (ancien: 4.0, un peu juste pour 2 bouteilles)

        # ── Timers internes ───────────────────────────────────────────────
        self._roundabout_timer    = 0.0
        self._obstacle_timer      = 0.0     # durée depuis détection obstacle
        self._avoid_timer         = 0.0     # durée du contournement
        self._curvature_count     = 0       # compteur de confirmation

        # ── Historique transitions ────────────────────────────────────────
        self.transitions          = []      # pour debug/log
        self._state_start_time    = time.time()

    # ─── Contrôles manuels ────────────────────────────────────────────────────
    def start(self):
        if self.state == State.IDLE:
            self._transition(State.LINE_FOLLOWING)

    def stop(self):
        if self.state != State.IDLE:
            self._transition(State.IDLE)

    def set_direction(self, direction: str):
        """Change la direction du rond-point à la volée."""
        assert direction in ('left', 'right'), "direction must be 'left' or 'right'"
        self.roundabout_direction = direction

    # ─── Mise à jour principale ───────────────────────────────────────────────
    def update(self, curvature: float, obstacle_dist: float, dt: float):
        """
        Appeler à chaque frame.
        curvature    : coeff |a| du polynôme de la voie
        obstacle_dist: distance frontale minimale LIDAR (m)
        dt           : delta time (s)
        """
        if self.state == State.IDLE:
            return

        obstacle_close = obstacle_dist < self.OBSTACLE_STOP_DIST
        obstacle_near  = obstacle_dist < self.OBSTACLE_AVOID_DIST

        # ── Arrêt d'urgence (priorité maximale) ───────────────────────────
        if obstacle_close and self.state not in (State.OBSTACLE_STOP,
                                                  State.OBSTACLE_AVOID,
                                                  State.CORRIDOR):
            if self.challenge == 1:
                self._transition(State.OBSTACLE_STOP)
                return
            elif self.challenge == 2:
                # On ne s'arrête pas, on contourne
                if self.state != State.OBSTACLE_AVOID:
                    self._avoid_timer = 0.0
                    self._transition(State.OBSTACLE_AVOID)
                return

        # ── OBSTACLE_STOP → attente dégagement ───────────────────────────
        if self.state == State.OBSTACLE_STOP:
            if not obstacle_close:
                self._obstacle_timer += dt
                if self._obstacle_timer > 0.8:  # attendre 0.8s que c'est dégagé
                    self._obstacle_timer = 0.0
                    self._transition(State.LINE_FOLLOWING)
            else:
                self._obstacle_timer = 0.0
            return

        # ── OBSTACLE_AVOID → contournement (challenge 2) ─────────────────
        if self.state == State.OBSTACLE_AVOID:
            self._avoid_timer += dt

            # Conditions de sortie :
            # 1. Timer minimum écoulé ET obstacle n'est plus "near"
            # 2. OU timer maximum écoulé (sécurité)
            if self._avoid_timer > self.AVOID_MAX_DURATION:
                # Timeout : on force le retour au suivi de ligne
                self._transition(State.LINE_FOLLOWING)
            elif self._avoid_timer > self.AVOID_MIN_DURATION and not obstacle_near:
                # L'obstacle est dépassé et on a attendu le minimum
                self._transition(State.LINE_FOLLOWING)
            return

        # ── LINE_FOLLOWING → détection rond-point ─────────────────────────
        if self.state == State.LINE_FOLLOWING:
            # Détection obstacle pour Challenge 2 (commence l'évitement
            # AVANT d'être trop proche)
            if self.challenge == 2 and obstacle_near and not obstacle_close:
                self._avoid_timer = 0.0
                self._transition(State.OBSTACLE_AVOID)
                return

            if curvature > self.CURVATURE_THRESHOLD:
                self._curvature_count += 1
                if self._curvature_count >= self.CURVATURE_CONFIRM_N:
                    self._curvature_count  = 0
                    self._roundabout_timer = 0.0
                    self._transition(State.ROUNDABOUT)
            else:
                # Décrémente au lieu de reset brutal (plus robuste au bruit)
                self._curvature_count = max(0, self._curvature_count - 1)
            return

        # ── ROUNDABOUT → timer ────────────────────────────────────────────
        if self.state == State.ROUNDABOUT:
            self._roundabout_timer += dt
            if self._roundabout_timer > self.ROUNDABOUT_DURATION:
                self._transition(State.LINE_FOLLOWING)
            return

    # ─── Propriétés booléennes ────────────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        return self.state not in (State.IDLE, State.OBSTACLE_STOP)

    @property
    def in_roundabout(self) -> bool:
        return self.state == State.ROUNDABOUT

    @property
    def is_avoiding(self) -> bool:
        return self.state == State.OBSTACLE_AVOID

    @property
    def time_in_state(self) -> float:
        return time.time() - self._state_start_time

    # ─── Helpers internes ─────────────────────────────────────────────────────
    def _transition(self, new_state: State):
        old = self.state.name
        self.state            = new_state
        self._state_start_time = time.time()
        entry = f"{old} → {new_state.name}"
        self.transitions.append(entry)
        if len(self.transitions) > 50:
            self.transitions.pop(0)

    def __str__(self):
        return (f"FSM[{self.state.name} | "
                f"dir={self.roundabout_direction} | "
                f"t={self.time_in_state:.1f}s]")

    def status_dict(self) -> dict:
        return {
            'state':     self.state.name,
            'direction': self.roundabout_direction,
            'challenge': self.challenge,
            't_in_state': round(self.time_in_state, 2),
        }
