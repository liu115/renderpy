#include <cstddef>
#include <iostream>
#include <sstream>
#include <GL/gl.h>
#include "opengl/egl.h"

namespace opengl {


void assertEGLError(const char* msg) {
	EGLint error = eglGetError();
    std::string m = msg;

	if (error != EGL_SUCCESS) {
		std::stringstream s;
		s << "EGL error at " << m << " with error code: " << error;
        s << " (0x" << std::hex << error << ")";
		throw std::runtime_error(s.str());
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
