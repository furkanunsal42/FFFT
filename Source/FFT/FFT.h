#pragma once

#include "Texture1D.h"
#include "Texture2D.h"
#include "Texture2DArray.h"
#include "Texture3D.h"
#include "Buffer.h"

#include "ComputeProgram.h"
#include "Tools/VariantedComputeProgram.h"

#include <filesystem>

namespace shader_directory {
	//extern std::filesystem::path ffft2_shader_directory;
	static std::filesystem::path ffft2_shader_directory = "../FFFT/Source/GLSL/FFT/";
}

class FFFT2 {
public:
	
	enum fft_dimension {
		x,
		y,
		z,
		xy,
		yz,
		xz,
		xyz,
	};

	template<typename T>
	constexpr static fft_dimension default_fft_dimension();

	enum windowing_function {
		hann,
		hamming,
	};

	enum fft_algorithm {
		fft_dft,
		fft_radix2_zeropadding,
		fft_radix2_dft,
		fft_radix235_zeropadding,
		fft_radix235_dft,
	};

	enum component {
		real,
		complex,
		real_complex,
	};

	template<typename T> void fft		(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), fft_algorithm algorithm = radix235_fft, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void i_fft		(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), fft_algorithm algorithm = radix235_fft, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	
	template<typename T> void shift		(T&	source,	T& target, glm::ivec3 shift_size, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void i_shift	(T& source, T& target, glm::ivec3 shift_size, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));

	template<typename T> void copy		(T& source,	T& target, component comp = real_complex, glm::ivec3 source_offset = glm::ivec3(0), glm::ivec3 target_offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> create	(T& source, component comp = real_complex, glm::ivec3 size = glm::ivec3(0));

	template<typename T> std::shared_ptr<T> pad		(T& source, component comp = real_complex, glm::ivec3 padding_size = glm::ivec3(0), glm::vec2 padding_value = glm::vec2(0), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void				i_pad	(T& source, T& target, component comp = real_complex, glm::ivec3 padding_size = glm::ivec3(0), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> i_pad	(T& source, component comp = real_complex, glm::vec2 padding_value = glm::vec2(0), glm::ivec3 padding_size = glm::ivec3(0), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));

	//void window();
	//void inverse_window();
	//void conjugate();
	//void multiply();
	//void divide();
	//void add();
	//void subtract();

	template<typename T> static bool is_complex(T& source);
	template<typename T> static bool is_real(T& source);
	template<typename T> static bool is_same(T& source, T& target);

private:

	std::string component_to_string(component component);

	enum data_type {
		buffer,
		texture1d,
		texture2d,
		texture2darray,
		texture3d,
	};

	void set_source_type(data_type data_type);
	void set_source_color_texture_format(Texture2D::ColorTextureFormat color_texture_format);

	void set_target_type(data_type data_type);
	void set_target_color_texture_format(Texture2D::ColorTextureFormat color_texture_format);

	std::vector<std::pair<std::string, std::string>> generate_shader_macros();
	void compile_shaders();
	
	Texture2D::ColorTextureFormat complex_texture_format(Texture2D::ColorTextureFormat real_texture);
	Texture2D::ColorTextureFormat real_texture_format(Texture2D::ColorTextureFormat complex_texture);

	template<typename T> void dft		(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix2	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix3	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix5	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	
	bool shaders_are_compiled = false;
	//data_type source_type = texture2d;
	//data_type target_type = texture2d;
	//Texture2D::ColorTextureFormat source_texture_format = Texture2D::ColorTextureFormat::RG16F;
	//Texture2D::ColorTextureFormat target_texture_format = Texture2D::ColorTextureFormat::RG16F;

	VariantedComputeProgram cp_dft;
	VariantedComputeProgram cp_copy;

	//std::shared_ptr<ComputeProgram> cp_dft_x;
	//std::shared_ptr<ComputeProgram> cp_dft_y;
	//std::shared_ptr<ComputeProgram> cp_dft_z;
	//std::shared_ptr<ComputeProgram> cp_copy_real;
	//std::shared_ptr<ComputeProgram> cp_copy_complex;
	//std::shared_ptr<ComputeProgram> cp_copy_real_complex;

};

#include "FFT_Templated.h"


//std::vector<std::pair<std::string, std::string>> FFFT2::generate_shader_macros()
//{
//	return std::vector<std::pair<std::string, std::string>>(
//		{
//			{ "ffft_source_format", T::ColorTextureFormat_to_OpenGL_compute_Image_format(source_texture_format)},
//			{ "ffft_target_format", T::ColorTextureFormat_to_OpenGL_compute_Image_format(target_texture_format)},
//
//			{ "source_image", T::ColorTextureFormat_to_OpenGL_compute_Image_type(source_texture_format)},
//			{ "target_image", T::ColorTextureFormat_to_OpenGL_compute_Image_type(target_texture_format)},
//
//			{ "source_image_dimensionality", T::get_texture_dimention()},
//			{ "target_image_dimensionality", T::get_texture_dimention()},
//
//			{ "source_number_type", T::ColorTextureFormat_channels(source_texture_format) == 1 ? "real" : "complex"},
//			{ "target_number_type", T::ColorTextureFormat_channels(target_texture_format) == 1 ? "real" : "complex"},
//		}
//		);
//}

void FFFT2::compile_shaders() {

	if (shaders_are_compiled) return;

	//cp_dft.begin_variant();
	//cp_dft.variant_define("ffft_source_format", Texture1D::ColorTextureFormat_to_OpenGL_compute_Image_format())
	//cp_dft.get_current_variant();

	//auto macros = generate_shader_macros();
	//
	//auto macros_x = macros;
	//auto macros_y = macros;
	//auto macros_z = macros;
	//
	//macros_x.push_back({ "ffft_axis", "axis_x" });
	//macros_y.push_back({ "ffft_axis", "axis_y" });
	//macros_z.push_back({ "ffft_axis", "axis_z" });

	//auto macros_copy_real = macros;
	//auto macros_copy_complex = macros;
	//auto macros_copy_real_complex = macros;

	//macros_copy_real.push_back({ "copy_operation", "real" });
	//macros_copy_complex.push_back({ "copy_operation", "complex" });
	//macros_copy_real_complex.push_back({ "copy_operation", "real_complex" });
	//
	//cp_dft_x = std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "dft.comp"), macros_x);
	//cp_dft_y = std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "dft.comp"), macros_y);
	//cp_dft_z = std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "dft.comp"), macros_z);
	//
	//cp_copy_real = std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "copy.comp"), macros_copy_real);
	//cp_copy_complex = std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "copy.comp"), macros_copy_complex);
	//cp_copy_real_complex = std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "copy.comp"), macros_copy_real_complex);
	//
	shaders_are_compiled = true;
}



template<typename T>
inline void FFFT2::copy(T& source, T& target, component comp, glm::ivec3 source_offset, glm::ivec3 target_offset,glm::ivec3 size)
{
	if (is_same(source, target)) {
		std::cout << "[FFFT Error] FFFT::copy() is called with identical source and target but self-copy is not supported" << std::endl;
		ASSERT(false);
		return;
	}
	
	if (size.x == 0) size.x = glm::max(to_ivec3(target.get_size()).x, 1);
	if (size.y == 0) size.y = glm::max(to_ivec3(target.get_size()).y, 1);
	if (size.z == 0) size.z = glm::max(to_ivec3(target.get_size()).z, 1);

	cp_copy.begin_variant();
	cp_copy.variant_define("ffft_source_format",	T::ColorTextureFormat_to_OpenGL_compute_Image_format(source.get_internal_format_color()));
	cp_copy.variant_define("ffft_target_format",	T::ColorTextureFormat_to_OpenGL_compute_Image_format(target.get_internal_format_color()));
	cp_copy.variant_define("source_image",			T::ColorTextureFormat_to_OpenGL_compute_Image_type(source.get_internal_format_color(), TextureBase2::get_texture_dimention<T>()));
	cp_copy.variant_define("target_image",			T::ColorTextureFormat_to_OpenGL_compute_Image_type(target.get_internal_format_color(), TextureBase2::get_texture_dimention<T>()));
	cp_copy.variant_define("source_image_dimensionality",		std::to_string(TextureBase2::get_texture_dimention<T>()));
	cp_copy.variant_define("target_image_dimensionality",		std::to_string(TextureBase2::get_texture_dimention<T>()));
	
	component source_component = 
		is_real(source)		? real : 
		is_complex(source)	? complex : real_complex;

	component target_component =
		is_real(target)		? real :
		is_complex(target)	? complex : real_complex;

	cp_copy.variant_define("source_number_type",	component_to_string(source_component));
	cp_copy.variant_define("target_number_type"	,	component_to_string(target_component));
	cp_copy.variant_define("copy_operation",		component_to_string(comp));
	
	ComputeProgram& kernel = *cp_copy.get_current_variant();

	kernel.update_uniform_as_image("fft_source_texture", source, 0);
	kernel.update_uniform_as_image("fft_target_texture", target, 0);

	kernel.update_uniform("fft_source_texture_resolution", source.get_size());
	kernel.update_uniform("fft_target_texture_resolution", target.get_size());
	
	kernel.update_uniform("fft_texture_source_offset", source_offset);
	kernel.update_uniform("fft_texture_target_offset", target_offset);
	kernel.update_uniform("fft_texture_region", size);

	kernel.dispatch_thread(size);
}