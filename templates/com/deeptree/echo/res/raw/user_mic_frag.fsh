#version 300 es

#define E 2.71828182846
#define pi 3.14159265358979323844

precision highp float;

struct ColoredSDF {
    float distance;
    vec4 color;
};

struct SDFArgs {
    vec2 st;
    float amount;
    float duration;
    float time;
    float mainRadius;
};

float scaled(float edge0, float edge1, float x) {
    return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}

float fixedSpring(float t, float d) {
    float s = mix((1.0 - (exp(-E * 2.0 * t) * cos((1.0 - d) * 115.0 * t))), 1.0, scaled(0.0, 1.0, t));
    return s * (1.0 - t) + t;
}

float opSmoothIntersection(float d1, float d2, float k) {
    if (k <= 0.0) {
        k = 0.000001;
    }
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

float opSmoothSubtraction(float d1, float d2, float k) {
    if (k <= 0.0) {
        k = 0.000001;
    }
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}

float sdRoundedBox(vec2 p, vec2 b, vec4 r) {
    r.xy = (p.x > 0.0) ? r.xy : r.zw;
    r.x = (p.y > 0.0) ? r.x : r.y;
    vec2 q = abs(p) - b + r.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r.x;
}

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float sdArc(vec2 p, vec2 sca, vec2 scb, float ra, float rb) {
    p *= mat2(sca.x, sca.y, -sca.y, sca.x);
    p.x = abs(p.x);
    return (scb.y * p.x > scb.x * p.y) ? length(p - ra * scb) - rb : abs(length(p) - ra) - rb;
}

float micSdf(vec2 st, float muted) {
    float d = 100.0;
    float strokeWidth = 0.03;
    vec2 elementSize = vec2(0.12, 0.26);
    vec2 elementPos = vec2(0.0, elementSize.y * 0.585);
    float element = sdRoundedBox(st - elementPos, elementSize, vec4(min(elementSize.x, elementSize.y)));
    element = mix(element - strokeWidth, abs(element) - strokeWidth * mix(1.0, 0.7, st.y / elementSize.y * 0.2), smoothstep(0.1, 0.8, muted));
    d = min(d, element);

    vec2 standSize = elementSize * 2.2;
    vec2 standPos = vec2(elementPos.x, elementPos.y - 0.05);
    st.y += 0.08;
    float ta = -pi / 2.0;// start angle
    float tb = pi / 2.0;// progress angle
    float w = 0.0;
    float stand = sdArc(st - standPos, vec2(sin(ta), cos(ta)), vec2(sin(tb), cos(tb)), standSize.x, w);
    stand = min(stand, sdSegment(st - standPos, vec2(standSize.x, 0.06), vec2(standSize.x, 0.0)));
    stand = min(stand, sdSegment(st - standPos, vec2(-standSize.x, 0.06), vec2(-standSize.x, 0.0)));

    float foot = sdSegment(st - standPos, vec2(0.0, -standSize.x), vec2(0.0, -standSize.x * 1.66));
    foot = min(foot, sdSegment(st - standPos, vec2(-standSize.x * 0.68, -standSize.x * 1.66), vec2(standSize.x * 0.68, -standSize.x * 1.66)));
    stand = min(stand, foot);

    vec2 slashFrom = vec2(-0.32, 0.43);
    vec2 slashTo = vec2(0.33, -0.22);
    float slashContainer = sdRoundedBox(st, vec2(slashTo.x, slashFrom.y) * 1.2, vec4(0.0));
    slashFrom = mix(slashFrom + normalize(slashFrom) * 0.3, slashFrom, muted);
    slashTo = mix(slashFrom, slashTo, muted);
    float slash = sdSegment(st, slashFrom, slashTo);
    slash = opSmoothIntersection(slash, slashContainer, 0.03);
    d = min(d, abs(stand) - strokeWidth);

    if (muted > 0.0) {
        d = opSmoothSubtraction(slash - strokeWidth * 2.8, d, 0.03);
        d = min(d, slash - strokeWidth);
    }

    return d;
}

ColoredSDF applyBottomAlignedBarsAndMicState(ColoredSDF sdf, SDFArgs args, vec4 avgMag, vec4 displayColor) {
    float d = 1000.0;
    int barCount = 5;
    int loopCount = barCount;
    if (args.amount == 0.0) {
        loopCount = 1;// optimization, only draw mic
    }
    for (int i = 0; i < loopCount; i++) {
        float f = float(float(i) + 0.5) / float(barCount);// fraction

        // height/width of bar
        float w = (1.0 / float(barCount)) * 0.42;
        float h = w;

        // wave-like motion when starting
        float entryDuration = 1.8;
        float entryAnimation = fixedSpring(scaled(0.0, entryDuration, args.duration), 0.94) * args.amount;
        vec2 pos = vec2((f) - 0.5, 0.0) * args.mainRadius * 1.9;
        pos.x *= entryAnimation;

        if (i == 0) {
            float micScale = mix(4.0, 6.0, args.amount);
            float yOffset = mix(w * 1.7, w * 2.0, args.amount);
            d = micSdf((args.st - pos + vec2(-w * 0.15 * args.amount, yOffset)) * micScale, 1.0 - args.amount) / micScale;
        } else {
            // audio level for bar
            h += avgMag[i - 1] * (0.1 + (1.0 - abs(f - 0.5) * 2.0) * 0.1) * 0.7;
            h = mix(w, h, smoothstep(0.8, 1.0, entryAnimation));

            float bubbleInDur = 0.5;
            float bubbleOutDur = 0.4;

            // tiny blobby wave on entry
            float bubbleEffect = fixedSpring(scaled(f / 4.0, f / 4.0 + bubbleInDur, args.duration - entryDuration / 8.0), 1.0) * pow(1.0 - scaled(f / 8.0 + bubbleInDur / 8.0, f / 4.0 + bubbleInDur / 8.0 + bubbleOutDur, args.duration - entryDuration / 8.0), 2.0);

            h += bubbleEffect * min(h, w);

            // hide when not in use
            w *= args.amount;
            h *= args.amount;

            h = min(h, 0.23); // never extend beyond container

            pos.y -= 0.25;
            pos.y += h;
            pos.y += bubbleEffect * w * 0.5;

            float dd = sdRoundedBox(args.st - pos, vec2(w, h), vec4(w));
            d = min(d, dd);
        }
    }

    sdf.distance = d;// no blending with previous sdfs here
    sdf.color = mix(sdf.color, displayColor, args.amount);
    return sdf;
}

// Fragment shader
in vec4 out_position;
in vec2 out_uv;

// Parameters
uniform float time;
uniform float stateListen;
uniform float listenTimestamp;
uniform float readyTimestamp;
uniform vec4 userMagnitudes;
uniform vec2 viewport;
uniform vec4 displayColor;

out vec4 fragColor;

// Fragment shader function
void main() {
    vec2 st = out_uv - 0.5;
    st.y *= viewport.y / viewport.x;

    vec2 microphoneST = st * vec2(1, -1) - vec2(0., 0.02);

    ColoredSDF micSdf;
    micSdf.color = vec4(pow(0.6, 2.2));// inactive color
    micSdf.distance = length(st);

    SDFArgs micArgs;
    micArgs.st = microphoneST;
    micArgs.amount = min(stateListen, 1.0);
    micArgs.mainRadius = 0.49;
    micArgs.time = time;
    micArgs.duration = time - listenTimestamp;

    float idleDuration = time - readyTimestamp;
    float micHidden = 1.0 - scaled(0.0, 0.3, idleDuration);

    micSdf = applyBottomAlignedBarsAndMicState(micSdf, micArgs, userMagnitudes, displayColor);

    // edge sharpness, higher value, blurrier border
    // practically free anti-aliasing
    float clampingTolerance = 0.005;

    vec4 micColor = smoothstep(clampingTolerance, 0.0, micSdf.distance) * (1.0 - micHidden) * micSdf.color;

    fragColor = micColor;
}
