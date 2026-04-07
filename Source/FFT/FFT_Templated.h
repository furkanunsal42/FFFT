#pragma once
#include "FFT.h"

//template<typename T>
//inline void FFFT2::fft(T& source, T& target, fft_dimension dimension, fft_algorithm algorithm, glm::ivec3 offset, glm::ivec3 size)
//{
//	dft(source, target, dimension, offset, size);
//}

//template<typename T>
//inline void FFFT2::dft(T& source, T& target, fft_dimension dimension, glm::ivec3 offset, glm::ivec3 size)
//{
//	if (dimension != x && dimension != y && dimension != z) {
//		std::cout << "[FFFT Error] void dft() is called with invalid dimension" << std::endl;
//		ASSERT(false);
//	}
//
//	ComputeProgram& kernel =
//		dimension == x ? *cp_dft_x :
//		dimension == y ? *cp_dft_y :
//		dimension == z ? *cp_dft_z : *cp_dft_x;
//
//	kernel.update_uniform_as_image("fft_read_texture", source, 0);
//	kernel.update_uniform_as_image("fft_write_texture", target, 0);
//
//	kernel.update_uniform("fft_texture_resolution", source.size());
//	kernel.update_uniform("fft_texture_offset", offset);
//	kernel.update_uniform("fft_texture_region", size);
//
//	kernel.dispatch_thread(size);
//}

template<> inline constexpr FFFT2::fft_dimension FFFT2::default_fft_dimension<Buffer>()			{ return x; }
template<> inline constexpr FFFT2::fft_dimension FFFT2::default_fft_dimension<Texture1D>()		{ return x; }
template<> inline constexpr FFFT2::fft_dimension FFFT2::default_fft_dimension<Texture2D>()		{ return xy; }
template<> inline constexpr FFFT2::fft_dimension FFFT2::default_fft_dimension<Texture2DArray>()	{ return xyz; }
template<> inline constexpr FFFT2::fft_dimension FFFT2::default_fft_dimension<Texture3D>()		{ return xyz; }


template<typename T>
inline bool FFFT2::is_same(T& source, T& target)
{
	return source.id == target.id;
}

namespace {

	glm::ivec3 to_ivec3(glm::ivec2 size2, int32_t blank_value = 0) {
		return glm::ivec3(size2.x, size2.y, blank_value);
	}

	glm::ivec3 to_ivec3(glm::ivec1 size1, int32_t blank_value = 0) {
		return glm::ivec3(size1.x, blank_value, blank_value);
	}

	template<typename texture_type>
	std::shared_ptr<texture_type> create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) = delete;

	template<> std::shared_ptr<Texture1D> create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture1D>(size.x, format, 1, 0);
	}
	template<> std::shared_ptr<Texture2D> create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture2D>(size.x, size.y, format, 1, 0);
	}
	template<> std::shared_ptr<Texture2DArray> create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture2DArray>(size.x, size.y, size.z, format, 1, 0);
	}
	template<> std::shared_ptr<Texture3D> create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture3D>(size.x, size.y, size.z, format, 1, 0);
	}

	template<typename texture_type, typename vector_type>
	void clear_texture_glm(texture_type& texture, vector_type offset, vector_type size, glm::vec4 color) = delete;

	template<> void clear_texture_glm(Texture1D& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
		texture.clear(color, offset.x, size.x, 0);
	}
	template<> void clear_texture_glm(Texture2D& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
		texture.clear(color, offset.x, offset.y, size.x, size.y, 0);
	}
	template<> void clear_texture_glm(Texture2DArray& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
		texture.clear(color, offset.x, offset.y, offset.z, size.x, size.y, size.z, 0);
	}
	template<> void clear_texture_glm(Texture3D& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
		texture.clear(color, offset.x, offset.y, offset.z, size.x, size.y, size.z, 0);
	}
}

template<typename T>
inline std::shared_ptr<T> FFFT2::create(T& source, component comp, glm::ivec3 size)
{
	compile_shaders();

	Texture2D::ColorTextureFormat format = source.get_internal_format_color();
	if (comp == real)			format = real_texture_format(format);
	if (comp == complex)		format = complex_texture_format(format);
	if (comp == real_complex)	format = complex_texture_format(format);

	if (size.x <= 0) size.x = to_ivec3(source.get_size()).x;
	if (size.y <= 0) size.y = to_ivec3(source.get_size()).y;
	if (size.z <= 0) size.z = to_ivec3(source.get_size()).z;

	return create_texture_glm<T>(to_ivec3(source.get_size()), format);
}


template<typename T>
inline void FFFT2::pad(T& source, T& target, glm::ivec3 offset, glm::vec2 padding_value)
{
	compile_shaders();

	if (glm::any(glm::lessThan(offset, glm::ivec3(0)))) {
		std::cout << "[FFFT Error] FFFT::pad() is called with an negative offset" << std::endl;
		ASSERT(false);
	}

	if (glm::any(glm::greaterThan(to_ivec3(source.get_size()) + offset, to_ivec3(target.get_size())))) {
		std::cout << "[FFFT Error] FFFT::pad() is called but specified offset + source's size exceeds target's size" << std::endl;
		ASSERT(false);
	}

	clear_texture_glm(target, glm::ivec3(0), to_ivec3(target.get_size()), glm::vec4(padding_value, 0, 1));
	copy(source, target, FFFT2::real_complex, glm::ivec3(0), offset, to_ivec3(source.get_size()));
}

template<typename T>
inline void FFFT2::i_pad(T& source, T& target, glm::ivec3 offset)
{
	compile_shaders();

	if (glm::any(glm::lessThan(offset, glm::ivec3(0)))) {
		std::cout << "[FFFT Error] FFFT::i_pad() is called with an negative offset" << std::endl;
		ASSERT(false);
	}

	if (glm::any(glm::greaterThan(to_ivec3(target.get_size()) + offset, to_ivec3(source.get_size())))) {
		std::cout << "[FFFT Error] FFFT::i_pad() is called but specified offset + target's size exceeds source's size" << std::endl;
		ASSERT(false);
	}

	copy(source, target, FFFT2::real_complex, offset, glm::ivec3(0), to_ivec3(target.get_size()));
}

template<typename T>
inline std::shared_ptr<T> FFFT2::pad(T& source, glm::ivec3 padded_size, glm::ivec3 offset, glm::vec2 padding_value)
{
	compile_shaders();

	if (glm::any(glm::lessThan(offset, glm::ivec3(0)))) {
		std::cout << "[FFFT Error] FFFT::pad() is called with negative size value" << std::endl;
		ASSERT(false);
	}
	
	if (glm::any(glm::lessThan(padded_size - offset, to_ivec3(source.get_size())))) {
		std::cout << "[FFFT Error] FFFT::pad() is called with invalid padded_size and offset values" << std::endl;
		ASSERT(false);
	}

	std::shared_ptr<T> target = create_texture_glm<T>(padded_size, source.get_internal_format_color());

	pad(source, *target, offset, padding_value);

	return target;
}

template<typename T>
inline std::shared_ptr<T> FFFT2::i_pad(T& source, glm::ivec3 offset, glm::ivec3 size)
{
	compile_shaders();

	if (glm::any(glm::lessThan(offset, glm::ivec3(0))) || glm::any(glm::lessThan(size, glm::ivec3(0)))) {
		std::cout << "[FFFT Error] FFFT::i_pad() is called with negative offset or size values" << std::endl;
		ASSERT(false);
	}

	std::shared_ptr<T> target = create_texture_glm<T>(size, source.get_internal_format_color());

	i_pad(source, *target, offset);

	return target;
}

template<typename T>
inline void FFFT2::shift(T& source, T& target, glm::ivec3 shift_amount)
{
	compile_shaders();

	if (source.get_size() != target.get_size()) {
		std::cout << "[FFFT Error] FFFT::shift() is called but given source and target sizes doesn't match" << std::endl;
		ASSERT(false);
	}

	if (is_same(source, target)) {
		std::cout << "[FFFT Error] FFFT::shift() is called with identical source and target but self-shift is not supported" << std::endl;
		ASSERT(false);
		return;
	}

	if (is_complex(source) && !is_complex(target) || is_real(source) && is_complex(target)) {
		std::cout << "[FFFT Error] FFFT::shift() is called with unmatching number systems (real vs complex)" << std::endl;
		ASSERT(false);
	}

	if (source.get_size() != target.get_size()) {
		std::cout << "[FFFT Error] FFFT::shift() is called with differently sized source and target" << std::endl;
		ASSERT(false);
	}

	cp_shift.begin_variant();
	cp_shift.variant_define("ffft_source_format",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(source.get_internal_format_color()));
	cp_shift.variant_define("ffft_target_format",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(target.get_internal_format_color()));
	cp_shift.variant_define("source_image",					TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(source.get_internal_format_color()));
	cp_shift.variant_define("target_image",					TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(target.get_internal_format_color()));
	cp_shift.variant_define("source_image_dimensionality",	std::to_string(TextureBase2::get_texture_dimention<T>()));
	cp_shift.variant_define("target_image_dimensionality",	std::to_string(TextureBase2::get_texture_dimention<T>()));

	ComputeProgram& kernel = *cp_shift.get_current_variant();

	glm::ivec3 total_size = to_ivec3(source.get_size(), 1);

	kernel.update_uniform_as_image("fft_source_texture", source, 0);
	kernel.update_uniform_as_image("fft_target_texture", target, 0);

	kernel.update_uniform("fft_texture_resolution", total_size);
	kernel.update_uniform("fft_shift_amount", shift_amount);

	kernel.dispatch_thread(total_size);
}

template<typename T>
inline std::shared_ptr<T> FFFT2::shift(T& source, glm::ivec3 shift_size)
{
	compile_shaders();
	std::shared_ptr<T> target = source.create_texture_with_same_parameters();
	shift(source, *target, shift_size);
	return target;
}

template<typename T>
inline void FFFT2::i_shift(T& source, T& target, glm::ivec3 shift_size)
{
	compile_shaders();
	shift(source, target, -shift_size);
}

template<typename T>
inline std::shared_ptr<T> FFFT2::i_shift(T& source, glm::ivec3 shift_size)
{
	compile_shaders();
	std::shared_ptr<T> target = source.create_texture_with_same_parameters();
	i_shift(source, *target, shift_size);
	return target;
}
