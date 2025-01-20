#include <cstddef>
#include <iostream>
#include <sstream>
#include <GL/gl.h>
#include "opengl/egl.h"

namespace opengl {


const char* eglErrorString(EGLint error) {
    switch (error) {
        case EGL_SUCCESS: return "EGL_SUCCESS: The last function succeeded without error.";
        case EGL_NOT_INITIALIZED: return "EGL_NOT_INITIALIZED: EGL is not initialized, or could not be initialized, for the specified EGL display connection.";
        case EGL_BAD_ACCESS: return "EGL_BAD_ACCESS: EGL cannot access a requested resource (for example a context is bound in another thread).";
        case EGL_BAD_ALLOC: return "EGL_BAD_ALLOC: EGL failed to allocate resources for the requested operation.";
        case EGL_BAD_ATTRIBUTE: return "EGL_BAD_ATTRIBUTE: An unrecognized attribute or attribute value was passed in the attribute list.";
        case EGL_BAD_CONTEXT: return "EGL_BAD_CONTEXT: An EGLContext argument does not name a valid EGL rendering context.";
        case EGL_BAD_CONFIG: return "EGL_BAD_CONFIG: An EGLConfig argument does not name a valid EGL frame buffer configuration.";
        case EGL_BAD_CURRENT_SURFACE: return "EGL_BAD_CURRENT_SURFACE: The current surface of the calling thread is a window, pixel buffer or pixmap that is no longer valid.";
        case EGL_BAD_DISPLAY: return "EGL_BAD_DISPLAY: An EGLDisplay argument does not name a valid EGL display connection.";
        case EGL_BAD_SURFACE: return "EGL_BAD_SURFACE: An EGLSurface argument does not name a valid surface (window, pixel buffer or pixmap) configured for GL rendering.";
        case EGL_BAD_MATCH: return "EGL_BAD_MATCH: Arguments are inconsistent (for example, a valid context requires buffers not supplied by a valid surface).";
        case EGL_BAD_PARAMETER: return "EGL_BAD_PARAMETER: One or more argument values are invalid.";
        case EGL_BAD_NATIVE_PIXMAP: return "EGL_BAD_NATIVE_PIXMAP: A NativePixmapType argument does not refer to a valid native pixmap.";
        case EGL_BAD_NATIVE_WINDOW: return "EGL_BAD_NATIVE_WINDOW: A NativeWindowType argument does not refer to a valid native window.";
        case EGL_CONTEXT_LOST: return "EGL_CONTEXT_LOST: A power management event has occurred. The application must destroy all contexts and reinitialise OpenGL ES state and objects to continue rendering.";
        default: return "Unknown EGL error.";
    }
}

void assertEGLError(const char* msg) {
    EGLint error = eglGetError();
    if (error != EGL_SUCCESS) {
        std::cerr << msg << ": " << eglErrorString(error) << std::endl;
        std::cerr << "Error code: " << error << " (0x" << std::hex << error << ")" << std::dec << std::endl;
        throw std::runtime_error("EGL error occurred.");
    }
}


EGLDisplay initEGL() {
    static const int MAX_DEVICES = 12;
    EGLDeviceEXT eglDevs[MAX_DEVICES];
    EGLint numDevices;
    // The return value
    EGLDisplay eglDpy;

    // Apply egl display without X-server
    // https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
    (PFNEGLQUERYDEVICESEXTPROC)
    eglGetProcAddress("eglQueryDevicesEXT");

    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
    (PFNEGLGETPLATFORMDISPLAYEXTPROC)
    eglGetProcAddress("eglGetPlatformDisplayEXT");

    eglQueryDevicesEXT(MAX_DEVICES, eglDevs, &numDevices);
    std::cout << "Detected " << numDevices << " devices" << std::endl;

    // Try all devices and select the first one that works
    EGLint major, minor;
    bool found = false;
    for (EGLint i = 0; i < numDevices; i++) {
        // std::cout << "Trying device " << i << std::endl;
        eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                    eglDevs[i], NULL);
        if (eglGetError() == EGL_SUCCESS && eglDpy != EGL_NO_DISPLAY) {
            // break;
            if (eglInitialize(eglDpy, &major, &minor) == EGL_TRUE) {
                std::cout << "Using device " << i << std::endl;
                std::cout << "Using EGL version " << major << "." << minor << std::endl;
                found = true;
                break;
            }
        }
    }

    assertEGLError("eglInitialize");
    if (!found) {
        throw std::runtime_error("Failed to initialize EGL");
    }

    // 2. Select an appropriate configuration
    EGLint numConfigs;
    EGLConfig eglCfg;

    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
    assertEGLError("eglChooseConfig");

    // 3. Create a surface
    EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg,
                                                pbufferAttribs);
    assertEGLError("eglCreatePbufferSurface");

    // 4. Bind the API
    eglBindAPI(EGL_OPENGL_API);
    assertEGLError("eglBindAPI");

    // 5. Create a context and make it current
    EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT,
                                         NULL);
    assertEGLError("eglCreateContext");

    eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);

    assertEGLError("eglMakeCurrent");
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "EGL version: " << eglQueryString(eglDpy, EGL_VERSION) << std::endl;

    return eglDpy;
}

void terminateEGL(EGLDisplay eglDpy) {
    eglTerminate(eglDpy);
}

} // namespace opengl
