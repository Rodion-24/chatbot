import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { RoundedBox } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

/**
 * A 3D robot head that turns to follow the cursor and blinks.
 *
 * Aiming is measured from this canvas's own position on screen, not from
 * the viewport centre: a head parked in a corner is not at the middle, so
 * a cursor at screen-centre genuinely lies off to one side of it.
 */

/** Distance from the head at which it is turned as far as it goes. */
const REACH = 560;
const YAW_MAX = 0.6;   // rad, left/right
const PITCH_MAX = 0.4; // rad, up/down
const EASE = 0.1;
const PUPIL_TRAVEL = 0.06;
const BLINK_MIN_MS = 2800;
const BLINK_MAX_MS = 4600;
const BLINK_MS = 140;

type Props = { className?: string };

/** Live direction from this canvas to the cursor, as [-1..1] per axis. */
function useCursorDirection() {
  const target = useRef(new THREE.Vector2(0, 0));
  const { gl } = useThree();

  useEffect(() => {
    const canvas = gl.domElement;
    const set = (x: number, y: number) => {
      const r = canvas.getBoundingClientRect();
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;
      target.current.set(
        THREE.MathUtils.clamp((x - cx) / REACH, -1, 1),
        THREE.MathUtils.clamp(-(y - cy) / REACH, -1, 1),
      );
    };
    const onMouse = (e: MouseEvent) => set(e.clientX, e.clientY);
    const onTouch = (e: TouchEvent) => {
      const t = e.touches[0];
      if (t) set(t.clientX, t.clientY);
    };
    window.addEventListener("mousemove", onMouse, { passive: true });
    window.addEventListener("touchmove", onTouch, { passive: true });
    return () => {
      window.removeEventListener("mousemove", onMouse);
      window.removeEventListener("touchmove", onTouch);
    };
  }, [gl]);

  return target;
}

function Eye({
  x,
  look,
  blink,
}: {
  x: number;
  look: React.RefObject<THREE.Vector2>;
  blink: React.RefObject<number>;
}) {
  const iris = useRef<THREE.Mesh>(null);
  const lid = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (iris.current) {
      iris.current.position.x = look.current.x * PUPIL_TRAVEL;
      iris.current.position.y = look.current.y * PUPIL_TRAVEL;
    }
    if (lid.current) {
      const s = Math.max(0.001, blink.current);
      lid.current.scale.y = s;
      lid.current.visible = s > 0.02;
    }
  });

  return (
    <group position={[x, 0.02, 0.34]}>
      {/* dark socket the iris sits in */}
      <mesh>
        <sphereGeometry args={[0.15, 24, 24]} />
        <meshStandardMaterial color="#1e293b" roughness={0.6} metalness={0.3} />
      </mesh>
      {/* emissive iris — stays legible at small on-screen sizes */}
      <mesh ref={iris} position={[0, 0, 0.085]}>
        <sphereGeometry args={[0.085, 20, 20]} />
        <meshStandardMaterial
          color="#38bdf8"
          emissive="#38bdf8"
          emissiveIntensity={1.3}
          roughness={0.2}
          toneMapped={false}
        />
      </mesh>
      <mesh position={[0.03, 0.035, 0.15]}>
        <sphereGeometry args={[0.018, 8, 8]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>
      {/* eyelid in body colour, squashed in Y to blink */}
      <mesh ref={lid} position={[0, 0, 0.012]} scale={[1, 0.001, 1]}>
        <sphereGeometry args={[0.165, 24, 24]} />
        <meshStandardMaterial color="#e2e8f0" roughness={0.35} metalness={0.45} />
      </mesh>
    </group>
  );
}

function Head() {
  const group = useRef<THREE.Group>(null);
  const cursor = useCursorDirection();
  const eased = useRef(new THREE.Vector2(0, 0));
  const blink = useRef(0);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let alive = true;
    let timer = 0;
    const schedule = () => {
      const gap = BLINK_MIN_MS + Math.random() * (BLINK_MAX_MS - BLINK_MIN_MS);
      timer = window.setTimeout(() => {
        if (!alive) return;
        const start = performance.now();
        const tick = () => {
          if (!alive) return;
          const t = (performance.now() - start) / BLINK_MS;
          // sin gives a smooth close-and-open in one pass.
          blink.current = t < 1 ? Math.sin(t * Math.PI) : 0;
          if (t < 1) requestAnimationFrame(tick);
          else schedule();
        };
        tick();
      }, gap);
    };
    schedule();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, []);

  useFrame((state) => {
    eased.current.lerp(cursor.current, EASE);
    if (group.current) {
      group.current.rotation.y = eased.current.x * YAW_MAX;
      group.current.rotation.x = -eased.current.y * PITCH_MAX;
      // A slow bob, applied to position only so it never fights the aim.
      group.current.position.y = Math.sin(state.clock.elapsedTime * 1.3) * 0.03;
    }
  });

  const body = useMemo(
    () => ({ color: "#e2e8f0", roughness: 0.35, metalness: 0.45 }),
    [],
  );
  const dark = useMemo(
    () => ({ color: "#475569", roughness: 0.4, metalness: 0.5 }),
    [],
  );

  return (
    <group ref={group}>
      <RoundedBox args={[1.05, 0.9, 0.85]} radius={0.3} smoothness={8}>
        <meshStandardMaterial {...body} />
      </RoundedBox>

      {/* dark faceplate */}
      <RoundedBox
        args={[0.82, 0.52, 0.06]}
        radius={0.18}
        smoothness={6}
        position={[0, 0.02, 0.42]}
      >
        <meshStandardMaterial {...dark} />
      </RoundedBox>

      <Eye x={-0.21} look={eased} blink={blink} />
      <Eye x={0.21} look={eased} blink={blink} />

      {/* mouth slot */}
      <mesh position={[0, -0.3, 0.43]}>
        <boxGeometry args={[0.3, 0.045, 0.02]} />
        <meshStandardMaterial color="#334155" />
      </mesh>

      {/* ear pods */}
      {[-0.56, 0.56].map((x) => (
        <mesh key={x} position={[x, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.12, 0.12, 0.09, 16]} />
          <meshStandardMaterial {...dark} />
        </mesh>
      ))}

      {/* antenna + light */}
      <mesh position={[0, 0.56, 0]}>
        <cylinderGeometry args={[0.018, 0.018, 0.24, 8]} />
        <meshStandardMaterial {...dark} />
      </mesh>
      <mesh position={[0, 0.72, 0]}>
        <sphereGeometry args={[0.075, 16, 16]} />
        <meshStandardMaterial
          color="#38bdf8"
          emissive="#38bdf8"
          emissiveIntensity={1.5}
          toneMapped={false}
        />
      </mesh>
    </group>
  );
}

export default function Robot3D({ className }: Props) {
  return (
    <div className={className} aria-hidden="true">
      <Canvas
        dpr={[1, 2]}
        camera={{ position: [0, 0, 2.6], fov: 34 }}
        gl={{ alpha: true, antialias: true }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[2.5, 3, 4]} intensity={1.5} />
        <directionalLight position={[-3, 1, -2]} intensity={0.9} color="#7dd3fc" />
        <directionalLight position={[0, -2, 2]} intensity={0.35} color="#e0f2fe" />
        <Head />
      </Canvas>
    </div>
  );
}
