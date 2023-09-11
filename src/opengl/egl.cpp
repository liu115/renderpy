#include <cstddef>
#include <iostream>
#include <sstream>
#include "opengl/egl.h"

namespace opengl {


void assertEGLError(const char* msg) {
	EGLint error = eglGetError();
    std::string m = msg;

	if (error != EGL_SUCCESS) {
		std::stringstream s;
		s << "EGL error at " << m << "with error code: " << error;
		throw std::runtime_error(s.str());
	}
}

EGLDisplay initEGL() {
    // Apply egl display without X-server
    // https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
    (PFNEGLQUERYDEVICESEXTPROC)
    eglGetProcAddress("eglQueryDevicesEXT");

    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
    (PFNEGLGETPLATFORMDISPLAYEXTPROC)
    eglGetProcAddress("eglGetPlatformDisplayEXT");

    static const int MAX_DEVICES = 4;
    EGLDeviceEXT eglDevs[MAX_DEVICES];
    EGLint numDevices;

    eglQueryDevicesEXT(MAX_DEVICES, eglDevs, &numDevices);

    EGLDisplay eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, 
                                        eglDevs[0], 0);

    // 1. Initialize EGL
    EGLint major, minor;
    std::cout << "Using EGL version " << major << "." << minor << std::endl;
    eglInitialize(eglDpy, &major, &minor);
    assertEGLError("eglInitialize");

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

    return eglDpy;
    // 6. Terminate EGL when finished
    // eglTerminate(eglDpy);
}

void terminateEGL(EGLDisplay eglDpy) {
    eglTerminate(eglDpy);
}

} // namespace opengl
