#pragma once
#include "FFT.h"

template<> inline constexpr FFFT2::fft_dimension FFFT2::default_fft_dimension<Buffer>()			{ return x; }
template<> inline constexpr FFFT2::fft_dimension FFFT2::default_fft_dimension<Texture1D>()		{ return x; }
template<> inline constexpr FFFT2::fft_dimension FFFT2::default_fft_dimension<Texture2D>()		{ return xy; }
template<> inline constexpr FFFT2::fft_dimension FFFT2::default_fft_dimension<Texture2DArray>()	{ return xyz; }
template<> inline constexpr FFFT2::fft_dimension FFFT2::default_fft_dimension<Texture3D>()		{ return xyz; }

template<typename T>
inline bool FFFT2::is_complex(T& source)
{
	return source.ColorTextureFormat_channels() == 2 && (source.ColorTextureFormat_to_Type() == Texture3D::Type::HALF_FLOAT || source.ColorTextureFormat_to_Type() == Texture3D::Type::FLOAT);;
}

template<typename T>
inline bool FFFT2::is_real(T& source)
{
	return source.ColorTextureFormat_channels() == 1 && (source.ColorTextureFormat_to_Type() == Texture3D::Type::HALF_FLOAT || source.ColorTextureFormat_to_Type() == Texture3D::Type::FLOAT);;
}

template<typename T>
inline bool FFFT2::is_same(T& source, T& target)
{
	return source.id == target.id;
}

template<typename T>
inline std::vector<std::pair<std::string, std::string>> FFFT2::generate_shader_macros()
{
	return std::vector<std::pair<std::string, std::string>>(
		{
			{ "ffft_source_format", T::ColorTextureFormat_to_OpenGL_compute_Image_format(source_texture_format)},
			{ "ffft_target_format", T::ColorTextureFormat_to_OpenGL_compute_Image_format(target_texture_format)},

			{ "source_image", T::ColorTextureFormat_to_OpenGL_compute_Image_type(source_texture_format)},
			{ "target_image", T::ColorTextureFormat_to_OpenGL_compute_Image_type(target_texture_format)},
			
			{ "source_image_dimensionality", T::get_texture_dimention()},
			{ "target_image_dimensionality", T::get_texture_dimention()},

			{ "source_number_type", T::ColorTextureFormat_channels(source_texture_format) == 1 ? "real" : "complex"},
			{ "target_number_type", T::ColorTextureFormat_channels(target_texture_format) == 1 ? "real" : "complex"},
		}
		);
}

template<typename T>
inline void FFFT2::compile_shaders() {

	if (shaders_are_compiled) return;

	auto macros = generate_shader_macros();
	
	auto macros_x = macros;
	auto macros_y = macros;
	auto macros_z = macros;

	macros_x.push_back({ "ffft_axis", "axis_x" });
	macros_y.push_back({ "ffft_axis", "axis_y" });
	macros_z.push_back({ "ffft_axis", "axis_z" });

	auto macros_copy_real			= macros;
	auto macros_copy_complex		= macros;
	auto macros_copy_real_complex	= macros;

	macros_copy_real		.push_back({ "copy_operation", "real"		 });
	macros_copy_complex		.push_back({ "copy_operation", "complex" 	 });
	macros_copy_real_complex.push_back({ "copy_operation", "real_complex"}); 

	cp_dft_x = std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "dft.comp"), macros_x);
	cp_dft_y = std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "dft.comp"), macros_y);
	cp_dft_z = std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "dft.comp"), macros_z);

	cp_copy_real			= std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "copy.comp"), macros_copy_real);
	cp_copy_complex			= std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "copy.comp"), macros_copy_complex);
	cp_copy_real_complex	= std::make_shared<ComputeProgram>(Shader(shader_directory::ffft2_shader_directory / "copy.comp"), macros_copy_real_complex);

	shaders_are_compiled = true;
}

namespace {
	template<typename texture_type, typename vector_type>
	std::shared_ptr<texture_type> create_texture_glm(vector_type size, Texture2D::ColorTextureFormat format) = delete;

	template<> std::shared_ptr<Texture1D> create_texture_glm(glm::ivec1 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture1D>(size.x, format, 1, 0);
	}
	template<> std::shared_ptr<Texture2D> create_texture_glm(glm::ivec2 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture2D>(size.x, size.y, format, 1, 0);
	}
	template<> std::shared_ptr<Texture2DArray> create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture2DArray>(size.x, size.y, size.z, format, 1, 0);
	}
	template<> std::shared_ptr<Texture3D> create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture3D>(size.x, size.y, size.z, format, 1, 0);
	}

	glm::ivec3 to_ivec3(glm::ivec2 size2) {
		return glm::ivec3(size2.x, size2.y, 0);
	}

	glm::ivec3 to_ivec3(glm::ivec1 size1) {
		return glm::ivec3(size1.x, 0, 0);
	}
}

template<typename T>
inline std::shared_ptr<T> FFFT2::create(T& source, component comp, glm::ivec3 size)
{
	Texture2D::ColorTextureFormat format = source.get_internal_format_color();
	if (comp == real_complex)	format = complex_texture_format(format);
	if (comp == real)			format = real_texture_format(format);
	if (comp == complex)		format = complex_texture_format(format);

	if (size.x <= 0) size.x = to_ivec3(source.get_size()).x;
	if (size.y <= 0) size.y = to_ivec3(source.get_size()).y;
	if (size.z <= 0) size.z = to_ivec3(source.get_size()).z;

	return create_texture_glm<T>(source.get_size(), format);
}

template<typename T>
inline void FFFT2::fft(T& source, T& target, fft_dimension dimension, fft_algorithm algorithm, glm::ivec3 offset, glm::ivec3 size)
{
	dft(source, target, dimension, offset, size);
}

template<typename T>
inline void FFFT2::dft(T& source, T& target, fft_dimension dimension, glm::ivec3 offset, glm::ivec3 size)
{
	if (dimension != x && dimension != y && dimension != z) {
		std::cout << "[FFFT Error] void dft() is called with invalid dimension" << std::endl;
		ASSERT(false);
	}

	ComputeProgram& kernel =
		dimension == x ? *cp_dft_x :
		dimension == y ? *cp_dft_y :
		dimension == z ? *cp_dft_z : *cp_dft_x;

	kernel.update_uniform_as_image("fft_read_texture", source, 0);
	kernel.update_uniform_as_image("fft_write_texture", target, 0);

	kernel.update_uniform("fft_texture_resolution", source.size());
	kernel.update_uniform("fft_texture_offset", offset);
	kernel.update_uniform("fft_texture_region", size);

	kernel.dispatch_thread(size);
}