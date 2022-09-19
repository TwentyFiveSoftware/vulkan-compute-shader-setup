compile_shader:
	glslangValidator -g0 -t -V --target-env vulkan1.3 -o cmake-build-debug/shader.comp.spv src/shaders/shader.comp
