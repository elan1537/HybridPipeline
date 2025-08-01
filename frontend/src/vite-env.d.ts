/// <reference types="vite/client" />

declare module '*.vs?raw' {
    const content: string;
    export default content;
}

declare module '*.fs?raw' {
    const content: string;
    export default content;
}

declare module '*.glsl?raw' {
    const content: string;
    export default content;
} 