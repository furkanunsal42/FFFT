#include "FFT.h"

//std::filesystem::path shader_directory::ffft2_shader_directory = "../FFFT/Source/GLSL/FFT/";

std::string FFFT2::component_to_string(component component)
{
	switch (component) {
		case real: 		   return "real";
		case complex: 	   return "complex";
		case real_complex: return "real_complex";
	}

	std::cout << "[FFFT Error] component_to_string() is called but given compoenent is invalid" << std::endl;
	ASSERT(false);
	return "";
}

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

void FFFT2::compile_shaders() {

	if (shaders_are_set) return;

	cp_copy.set_shader(		Shader(shader_directory::ffft2_shader_directory / "copy.comp"));
	cp_shift.set_shader(	Shader(shader_directory::ffft2_shader_directory / "shift.comp"));
	cp_dft.set_shader(		Shader(shader_directory::ffft2_shader_directory / "dft.comp"));

	shaders_are_set = true;

}