uniform sampler2D localColorSampler;
uniform sampler2D localDepthSampler;
uniform sampler2D wsColorSampler;
uniform sampler2D wsDepthSampler;
uniform bool wsFlipX;
uniform bool fusionFlipX;
uniform float contrast;
uniform float brightness;
varying vec2 vUv;

void main() {
    // 기본 UV 계산
    vec2 currentUv = vUv;

    // fusionFlipX가 true이면 UV를 플립해서 모든 계산을 다시 수행
    if (fusionFlipX) {
        currentUv.x = 1.0 - currentUv.x;
    }

    vec2 localUv = currentUv;
    vec2 wsUv = vec2(currentUv.x, 1.0 - currentUv.y);

    // wsFlipX가 true일 때만 wsUv의 X축을 추가로 flip
    if (wsFlipX) {
        wsUv.x = 1.0 - wsUv.x;
    }

    vec4 localColor = texture2D(localColorSampler, localUv);
    float localDepth = texture2D(localDepthSampler, localUv).r;

    float d_8bit = floor(localDepth * 255.0) / 255.0;

    vec4 wsColor = texture2D(wsColorSampler, wsUv);
    float wsDepth = texture2D(wsDepthSampler, wsUv).r;

    bool useLocal = wsDepth > d_8bit;

    vec4 finalColor = useLocal ? localColor : wsColor;

    // vec3 adjustedColor = (finalColor.rgb - 0.5) * contrast + 0.5;
    // adjustedColor = adjustedColor * brightness;

    gl_FragColor = linearToOutputTexel(finalColor);
}