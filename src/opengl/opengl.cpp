#include "opengl/opengl.h"

namespace opengl {
bool init() {
    // Init opengl
    GLenum err;
    glewExperimental = true;    // Needed for core profile
	if ((err = glewInit()) != GLEW_OK) {
        std::cerr << glewGetErrorString(err) << std::endl;
		std::cerr << "Failed to initialize GLEW" << std::endl;
		return false;
	}
    return true;
}
};
