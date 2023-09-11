/*  
    https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/
*/
#pragma once
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>


namespace opengl {
static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
};    

static const int pbufferWidth = 9;
static const int pbufferHeight = 9;

static const EGLint pbufferAttribs[] = {
    EGL_WIDTH, pbufferWidth,
    EGL_HEIGHT, pbufferHeight,
    EGL_NONE,
};
EGLDisplay initEGL();
void terminateEGL(EGLDisplay eglDpy);

} // namespace opengl
