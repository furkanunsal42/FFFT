#include "FFT.h"

//std::filesystem::path shader_directory::ffft2_shader_directory = "../FFFT/Source/GLSL/FFT/";

void FFFT2::set_source_type(data_type data_type){
	if (source_type == data_type) return;
	source_type = data_type;
	shaders_are_compiled = false;
}

void FFFT2::set_source_color_texture_format(Texture2D::ColorTextureFormat color_texture_format){
	if (source_texture_format == color_texture_format) return;
	source_texture_format = color_texture_format;
	shaders_are_compiled = false;
}


void FFFT2::set_target_type(data_type data_type){
	if (target_type == data_type) return;
	target_type = data_type;
	shaders_are_compiled = false;
}

void FFFT2::set_target_color_texture_format(Texture2D::ColorTextureFormat color_texture_format){
	if (target_texture_format == color_texture_format) return;
	target_texture_format = color_texture_format;
	shaders_are_compiled = false;
}

