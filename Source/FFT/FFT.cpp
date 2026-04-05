#include "FFT.h"

//std::filesystem::path shader_directory::ffft2_shader_directory = "../FFFT/Source/GLSL/FFT/";

std::string FFFT2::component_to_string(component component)
{
	switch (component) {
		case x: return "x";
		case y: return "y";
		case z: return "z";
		case xy: return "xy";
		case yz: return "yz";
		case xz: return "xz";
		case xyz: return "xyz";
	}

	std::cout << "[FFFT Error] component_to_string() is called but given compoenent is invalid" << std::endl;
	ASSERT(false);
	return "";
}

//void FFFT2::set_source_type(data_type data_type){
//	if (source_type == data_type) return;
//	source_type = data_type;
//	shaders_are_compiled = false;
//}
//
//void FFFT2::set_source_color_texture_format(Texture2D::ColorTextureFormat color_texture_format){
//	if (source_texture_format == color_texture_format) return;
//	source_texture_format = color_texture_format;
//	shaders_are_compiled = false;
//}
//
//
//void FFFT2::set_target_type(data_type data_type){
//	if (target_type == data_type) return;
//	target_type = data_type;
//	shaders_are_compiled = false;
//}
//
//void FFFT2::set_target_color_texture_format(Texture2D::ColorTextureFormat color_texture_format){
//	if (target_texture_format == color_texture_format) return;
//	target_texture_format = color_texture_format;
//	shaders_are_compiled = false;
//}

Texture2D::ColorTextureFormat FFFT2::complex_texture_format(Texture2D::ColorTextureFormat real_texture){
	switch (real_texture) {
	case Texture2D::ColorTextureFormat::R32F: return Texture2D::ColorTextureFormat::RG32F;
	case Texture2D::ColorTextureFormat::RG32F: return Texture2D::ColorTextureFormat::RG32F;
	case Texture2D::ColorTextureFormat::R16F: return Texture2D::ColorTextureFormat::RG16F;
	case Texture2D::ColorTextureFormat::RG16F: return Texture2D::ColorTextureFormat::RG16F;
	}

	std::cout << "[FFFT Error] FFFT::complex_texture_format() is called but given original format is not real" << std::endl;
	ASSERT(false);
	return Texture2D::ColorTextureFormat::RG16F;
}

Texture2D::ColorTextureFormat FFFT2::real_texture_format(Texture2D::ColorTextureFormat complex_texture){
	switch (complex_texture) {
	case Texture2D::ColorTextureFormat::RG32F: return Texture2D::ColorTextureFormat::R32F;
	case Texture2D::ColorTextureFormat::R32F: return Texture2D::ColorTextureFormat::R32F;
	case Texture2D::ColorTextureFormat::RG16F: return Texture2D::ColorTextureFormat::R16F;
	case Texture2D::ColorTextureFormat::R16F: return Texture2D::ColorTextureFormat::R16F;
	}

	std::cout << "[FFFT Error] FFFT::real_texture_format() is called but given original format is not complex" << std::endl;
	ASSERT(false);
	return Texture2D::ColorTextureFormat::R16F;
}
