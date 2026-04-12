#include "FFT.h"
#include <functional>

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
	cp_split.set_shader(	Shader(shader_directory::ffft2_shader_directory / "split.comp"));
	cp_step.set_shader(		Shader(shader_directory::ffft2_shader_directory / "step.comp"));
	cp_op.set_shader(		Shader(shader_directory::ffft2_shader_directory / "op.comp"));

	shaders_are_set = true;

}

FFFT2::fft_plan FFFT2::create_plan(size_t array_size, const std::vector<size_t>& supported_radixes)
{
	fft_plan plan;

	do {
		fft_iteration iteration;

		size_t found_radix = fft_iteration::radix_dft;

		for (size_t radix : supported_radixes) {
			if (array_size % radix == 0) {
				found_radix = radix;
				array_size /= radix;
				break;
			}
		}

		iteration.radix = found_radix;
		iteration.chunk_size = array_size;
		plan.iterations.push_back(iteration);
		
		if (iteration.chunk_size == 1 || found_radix == fft_iteration::radix_dft)
			break;
		
	} while (true);

	return plan;
}

FFFT2::fft_plan FFFT2::create_plan(size_t array_size, size_t supported_max_radix, size_t supported_min_radix)
{
	if (supported_max_radix < supported_min_radix) {
		ASSERT(false);
	}

	fft_plan plan;

	do {
		fft_iteration iteration;

		size_t found_radix = fft_iteration::radix_dft;

		for (size_t radix = supported_max_radix; radix >= supported_min_radix; radix--) {
			if (array_size % radix == 0) {
				found_radix = radix;
				array_size /= radix;
				break;
			}
		}

		iteration.radix = found_radix;
		iteration.chunk_size = array_size;
		plan.iterations.push_back(iteration);

		if (iteration.chunk_size == 1 || found_radix == fft_iteration::radix_dft)
			break;

	} while (true);

	return plan;
}
