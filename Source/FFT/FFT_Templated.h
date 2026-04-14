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
	return source.ColorTextureFormat_channels(source.get_internal_format_color()) == 2 && (source.ColorTextureFormat_to_Type(source.get_internal_format_color()) == Texture3D::Type::HALF_FLOAT || source.ColorTextureFormat_to_Type(source.get_internal_format_color()) == Texture3D::Type::FLOAT);;
}

template<typename T>
inline bool FFFT2::is_real(T& source)
{
	return source.ColorTextureFormat_channels(source.get_internal_format_color()) == 1 && (source.ColorTextureFormat_to_Type(source.get_internal_format_color()) == Texture3D::Type::HALF_FLOAT || source.ColorTextureFormat_to_Type(source.get_internal_format_color()) == Texture3D::Type::FLOAT);;
}

template<typename T>
inline bool FFFT2::is_same(T& source, T& target)
{
	return source.id == target.id;
}

template<> 
inline std::shared_ptr<Texture1D> FFFT2::create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
	return std::make_shared<Texture1D>(size.x, format, 1, 0);
}
template<> 
inline std::shared_ptr<Texture2D> FFFT2::create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
	return std::make_shared<Texture2D>(size.x, size.y, format, 1, 0);
}
template<> 
inline std::shared_ptr<Texture2DArray> FFFT2::create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
	return std::make_shared<Texture2DArray>(size.x, size.y, size.z, format, 1, 0);
}
template<> 
inline std::shared_ptr<Texture3D> FFFT2::create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
	return std::make_shared<Texture3D>(size.x, size.y, size.z, format, 1, 0);
}

template<> 
inline void FFFT2::clear_texture_glm(Texture1D& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
	texture.clear(color, offset.x, size.x, 0);
}
template<> 
inline void FFFT2::clear_texture_glm(Texture2D& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
	texture.clear(color, offset.x, offset.y, size.x, size.y, 0);
}
template<> 
inline void FFFT2::clear_texture_glm(Texture2DArray& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
	texture.clear(color, offset.x, offset.y, offset.z, size.x, size.y, size.z, 0);
}
template<> 
inline void FFFT2::clear_texture_glm(Texture3D& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
	texture.clear(color, offset.x, offset.y, offset.z, size.x, size.y, size.z, 0);
}

template<typename T>
inline  void FFFT2::op(T& source, glm::vec4 constant, std::string operation, std::string additional_definition)
{
	compile_shaders();

	if (!is_complex(source) && !is_real(source)) {
		std::cout << "[FFFT Error] FFFT::op() is called with a source that is neither of real or complex type" << std::endl;
		ASSERT(false);
	}

	cp_op.begin_variant();
	cp_op.variant_define("ffft_source_format",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(source.get_internal_format_color()));
	cp_op.variant_define("source_image",				TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(source.get_internal_format_color()));
	cp_op.variant_define("source_image_dimensionality", std::to_string(TextureBase2::get_texture_dimention<T>()));

	cp_op.variant_define("op",	operation);
	cp_op.variant_define("def", std::string("uniform vec4 constant;"));

	ComputeProgram& kernel = *cp_op.get_current_variant();

	kernel.update_uniform_as_image("fft_source_texture", source, 0);
	kernel.update_uniform("fft_texture_resolution", to_ivec3(source.get_size(), 1));
	kernel.update_uniform("constant", constant);

	kernel.dispatch_thread(to_ivec3(source.get_size(), 1));
}

template<typename T>
inline void FFFT2::conjugate(T& source)
{
	op(source, glm::vec4(0), "value.y *= -1");
}

template<typename T>
inline void FFFT2::multiply(T& source, glm::vec2 constant)
{
	op(source, glm::vec4(constant, 1, 1), "value.xy *= constant.xy");
}

template<typename T>
inline void FFFT2::divide(T& source, glm::vec2 constant)
{
	constant = glm::max(constant, glm::vec2(0.00001));
	op(source, glm::vec4(constant, 1, 1), "value.xy /= constant.xy");
}

template<typename T>
inline void FFFT2::add(T& source, glm::vec2 constant)
{
	op(source, glm::vec4(constant, 0, 0), "value.xy += constant.xy");
}

template<typename T>
inline void FFFT2::subtract(T& source, glm::vec2 constant)
{
	op(source, glm::vec4(constant, 0, 0), "value.xy -= constant.xy");
}

template<typename T>
inline void FFFT2::taper_tukey(T& source, fft_dimension dimension, float alpha)
{
	compile_shaders();

	if (dimension != x && dimension != y && dimension != z) {
		std::cout << "[FFFT Error] FFFT::taper_tukey() is called with composite dimension but it is not supported" << std::endl;
		ASSERT(false);
	}

	cp_window.begin_variant();
	cp_window.variant_define("ffft_source_format",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(source.get_internal_format_color()));
	cp_window.variant_define("source_image",				TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(source.get_internal_format_color()));
	cp_window.variant_define("source_image_dimensionality", std::to_string(TextureBase2::get_texture_dimention<T>()));
	
	cp_step.variant_define("direction",
		dimension == x ? "axis_x" :
		dimension == y ? "axis_y" :
		dimension == z ? "axis_z" : "axis_x");

	ComputeProgram& kernel = *cp_window.get_current_variant();

	kernel.update_uniform_as_image("fft_source_texture", source, 0);
	kernel.update_uniform("fft_texture_resolution", to_ivec3(source.get_size(), 1));
	kernel.update_uniform("alpha", alpha);

	kernel.dispatch_thread(to_ivec3(source.get_size(), 1));
}

template<typename T>
inline void FFFT2::split(T& source, T& target, glm::ivec3 split_count, glm::ivec3 group_count)
{
	if (is_same(source, target)) {
		std::shared_ptr<T> texture = source.create_texture_with_same_parameters();
		copy(source, *texture, real_complex);
		split(*texture, source, split_count, group_count);
		return;
	}

	compile_shaders();

	if (is_same(source, target)) {
		ASSERT(false);
	}

	if (source.get_size() != target.get_size()) {
		ASSERT(false);
	}

	if (source.get_internal_format_color() != target.get_internal_format_color()) {
		ASSERT(false);
	}

	if (glm::any(glm::lessThan(group_count, glm::ivec3(1)))) {
		ASSERT(false);
	}

	if (glm::any(glm::notEqual(to_ivec3(source.get_size(), 1) % group_count, glm::ivec3(0)))) {
		ASSERT(false);
	}

	cp_split.begin_variant();
	cp_split.variant_define("ffft_source_format",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(source.get_internal_format_color()));
	cp_split.variant_define("ffft_target_format",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(target.get_internal_format_color()));
	cp_split.variant_define("source_image",					TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(source.get_internal_format_color()));
	cp_split.variant_define("target_image",					TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(target.get_internal_format_color()));
	cp_split.variant_define("source_image_dimensionality",	std::to_string(TextureBase2::get_texture_dimention<T>()));
	cp_split.variant_define("target_image_dimensionality",	std::to_string(TextureBase2::get_texture_dimention<T>()));

	ComputeProgram& kernel = *cp_split.get_current_variant();

	kernel.update_uniform_as_image("fft_source_texture", source, 0);
	kernel.update_uniform_as_image("fft_target_texture", target, 0);

	kernel.update_uniform("fft_source_texture_resolution", to_ivec3(source.get_size(), 1));
	kernel.update_uniform("fft_target_texture_resolution", to_ivec3(target.get_size(), 1));

	kernel.update_uniform("fft_texture_source_offset", glm::ivec3(0));
	kernel.update_uniform("fft_texture_target_offset", glm::ivec3(0));
	kernel.update_uniform("fft_texture_region", to_ivec3(source.get_size(), 1));

	kernel.update_uniform("group_count", group_count);
	kernel.update_uniform("split_count", split_count);

	kernel.dispatch_thread(to_ivec3(source.get_size(), 1));
}


template<typename T>
inline void FFFT2::step(T& source, T& target, size_t radix, fft_dimension dimension, bool inverse, glm::ivec3 group_count)
{
	compile_shaders();

	if (radix == fft_iteration::radix_dft) {
		dft(source, target, dimension, inverse);
		return;
	}

	if (is_same(source, target)) {
		ASSERT(false);
	}

	if (source.get_size() != target.get_size()) {
		ASSERT(false);
	}

	if (source.get_internal_format_color() != target.get_internal_format_color()) {
		ASSERT(false);
	}

	if (radix == 0) {
		ASSERT(false);
	}

	if (dimension != x && dimension != y && dimension != z) {
		ASSERT(false);
	}

	if (
		(dimension == x && to_ivec3(source.get_size(), 1).x % radix != 0) ||
		(dimension == y && to_ivec3(source.get_size(), 1).y % radix != 0) ||
		(dimension == z && to_ivec3(source.get_size(), 1).z % radix != 0)
		) {
		ASSERT(false);
	}

	cp_step.begin_variant();
	cp_step.variant_define("ffft_source_format",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(source.get_internal_format_color()));
	cp_step.variant_define("ffft_target_format",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(target.get_internal_format_color()));
	cp_step.variant_define("source_image",					TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(source.get_internal_format_color()));
	cp_step.variant_define("target_image",					TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(target.get_internal_format_color()));
	cp_step.variant_define("source_image_dimensionality",	std::to_string(TextureBase2::get_texture_dimention<T>()));
	cp_step.variant_define("target_image_dimensionality",	std::to_string(TextureBase2::get_texture_dimention<T>()));

	cp_step.variant_define("radix", std::to_string(radix));
	cp_step.variant_define("direction", 
		dimension == x ? "axis_x" :
		dimension == y ? "axis_y" :
		dimension == z ? "axis_z" : "axis_x"
	);
	cp_step.variant_define("fft_mode", inverse ? "fft_inverse" : "fft_forward");

	ComputeProgram& kernel = *cp_step.get_current_variant();

	kernel.update_uniform_as_image("fft_source_texture", source, 0);
	kernel.update_uniform_as_image("fft_target_texture", target, 0);

	kernel.update_uniform("fft_texture_resolution", to_ivec3(source.get_size(), 1));
	kernel.update_uniform("group_count", group_count);

	glm::ivec3 dispatch_size = to_ivec3(source.get_size(), 1);
	if (dimension == x) dispatch_size.x /= radix;
	if (dimension == y) dispatch_size.y /= radix;
	if (dimension == z) dispatch_size.z /= radix;

	kernel.dispatch_thread(dispatch_size);
}

template<typename T>
inline void FFFT2::dft(T& source, T& target, fft_dimension dimension, bool inverse, glm::ivec3 group_count)
{
	compile_shaders();

	if (is_same(source, target)) {
		std::cout << "[FFFT Error] FFFT::dft() is called but same texture cannot be used for source and target at the same time" << std::endl;
		ASSERT(false);
	}

	if (source.get_size() != target.get_size()) {
		std::cout << "[FFFT Error] FFFT::dft() is called but source and texture sizes doesn't match" << std::endl;
		ASSERT(false);
	}

	if (source.get_internal_format_color() != target.get_internal_format_color()) {
		std::cout << "[FFFT Error] FFFT::dft() is called but source and target internal formats doesn't match" << std::endl;
		ASSERT(false);
	}

	if (dimension != x && dimension != y && dimension != z) {
		std::cout << "[FFFT Error] FFFT::dft() is called but only a single dimension must be selected" << std::endl;
		ASSERT(false);
	}

	if (
		(dimension == x && to_ivec3(source.get_size(), 1).x < group_count.x) ||
		(dimension == y && to_ivec3(source.get_size(), 1).y < group_count.y) ||
		(dimension == z && to_ivec3(source.get_size(), 1).z < group_count.z)
		) {
		std::cout << "[FFFT Error] FFFT::dft() is called but group_count exceeds the texture resolution" << std::endl;
		ASSERT(false);
	}

	cp_dft.begin_variant();
	cp_dft.variant_define("ffft_source_format",				TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(source.get_internal_format_color()));
	cp_dft.variant_define("ffft_target_format",				TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(target.get_internal_format_color()));
	cp_dft.variant_define("source_image",					TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(source.get_internal_format_color()));
	cp_dft.variant_define("target_image",					TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(target.get_internal_format_color()));
	cp_dft.variant_define("source_image_dimensionality",	std::to_string(TextureBase2::get_texture_dimention<T>()));
	cp_dft.variant_define("target_image_dimensionality",	std::to_string(TextureBase2::get_texture_dimention<T>()));

	cp_dft.variant_define("direction", 
		dimension == x ? "axis_x" :
		dimension == y ? "axis_y" :
		dimension == z ? "axis_z" : "axis_x"
	);
	cp_dft.variant_define("fft_mode", inverse ? "fft_inverse" : "fft_forward");

	ComputeProgram& kernel = *cp_dft.get_current_variant();

	kernel.update_uniform_as_image("fft_source_texture", source, 0);
	kernel.update_uniform_as_image("fft_target_texture", target, 0);

	kernel.update_uniform("fft_texture_resolution", to_ivec3(source.get_size(), 1));
	kernel.update_uniform("group_count", group_count);

	kernel.dispatch_thread(to_ivec3(source.get_size(), 1));

	float divisor =
		dimension == x ? glm::sqrt(to_ivec3(target.get_size(), 1).x) :
		dimension == y ? glm::sqrt(to_ivec3(target.get_size(), 1).y) :
		dimension == z ? glm::sqrt(to_ivec3(target.get_size(), 1).z) : 1;

	//divide(target, glm::vec2(divisor));
}


template<typename T>
inline void FFFT2::copy(T& source, T& target, component comp, glm::ivec3 source_offset, glm::ivec3 target_offset, glm::ivec3 size)
{
	compile_shaders();

	if (is_same(source, target)) {
		std::cout << "[FFFT Error] FFFT::copy() is called with identical source and target but self-copy is not supported" << std::endl;
		ASSERT(false);
		return;
	}

	if (!is_complex(source) && !is_real(source)) {
		std::cout << "[FFFT Error] FFFT::copy() is called with a source that is neither of real or complex type" << std::endl;
		ASSERT(false);
	}

	if (!is_complex(target) && !is_real(target)) {
		std::cout << "[FFFT Error] FFFT::copy() is called with a target that is neither of real or complex type" << std::endl;
		ASSERT(false);
	}

	if (is_real(source) && is_real(target) && comp == real)
		comp = real_complex;

	if (is_real(source) && is_real(target) && comp == complex)
		return;
	
	if (size.x == 0) size.x = glm::min(to_ivec3(source.get_size(), 1).x - source_offset.x, to_ivec3(target.get_size(), 1).x - target_offset.x);
	if (size.y == 0) size.y = glm::min(to_ivec3(source.get_size(), 1).y - source_offset.y, to_ivec3(target.get_size(), 1).y - target_offset.y);
	if (size.z == 0) size.z = glm::min(to_ivec3(source.get_size(), 1).z - source_offset.z, to_ivec3(target.get_size(), 1).z - target_offset.z);

	bool source_overflow = glm::any(glm::greaterThan(source_offset + size, to_ivec3(source.get_size(), 1)));
	bool target_overflow = glm::any(glm::greaterThan(target_offset + size, to_ivec3(target.get_size(), 1)));

	if (source_overflow || target_overflow) {
		std::cout << "[FFFT Error] FFFT::copy() is called but specified offset and size exceeds data size" << std::endl;
		ASSERT(false);
		return;
	}

	cp_copy.begin_variant();
	cp_copy.variant_define("ffft_source_format",	TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(source.get_internal_format_color()));
	cp_copy.variant_define("ffft_target_format",	TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(target.get_internal_format_color()));
	cp_copy.variant_define("source_image",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(source.get_internal_format_color()));
	cp_copy.variant_define("target_image",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(target.get_internal_format_color()));
	cp_copy.variant_define("source_image_dimensionality", std::to_string(TextureBase2::get_texture_dimention<T>()));
	cp_copy.variant_define("target_image_dimensionality", std::to_string(TextureBase2::get_texture_dimention<T>()));

	cp_copy.variant_define("copy_operation", component_to_string(comp));

	ComputeProgram& kernel = *cp_copy.get_current_variant();

	kernel.update_uniform_as_image("fft_source_texture", source, 0);
	kernel.update_uniform_as_image("fft_target_texture", target, 0);

	kernel.update_uniform("fft_source_texture_resolution", to_ivec3(source.get_size(), 1));
	kernel.update_uniform("fft_target_texture_resolution", to_ivec3(target.get_size(), 1));

	kernel.update_uniform("fft_texture_source_offset", source_offset);
	kernel.update_uniform("fft_texture_target_offset", target_offset);
	kernel.update_uniform("fft_texture_region", size);

	kernel.dispatch_thread(size);
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

	return create_texture_glm<T>(size, format);
}

template<typename T>
inline std::shared_ptr<T> FFFT2::create(TextureBase2::ColorTextureFormat format, component comp, glm::ivec3 size)
{
	compile_shaders();

	if (comp == real)			format = real_texture_format(format);
	if (comp == complex)		format = complex_texture_format(format);
	if (comp == real_complex)	format = complex_texture_format(format);

	return create_texture_glm<T>(size, format);
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
	if (is_same(source, target)) {
		std::shared_ptr<T> texture = shift(source, shift_amount);
		copy(*texture, source, real_complex);
		return;
	}

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
inline void FFFT2::shift(T& source, T& target, fft_dimension dimension)
{
	glm::ivec3 shift_amount = glm::ivec3(0);
	if (dimension & x) shift_amount.x = to_ivec3(source.get_size(), 1).x / 2;
	if (dimension & y) shift_amount.y = to_ivec3(source.get_size(), 1).y / 2;
	if (dimension & z) shift_amount.z = to_ivec3(source.get_size(), 1).z / 2;
	shift(source, target, shift_amount);
}

template<typename T>
inline std::shared_ptr<T> FFFT2::shift(T& source, fft_dimension dimension)
{
	glm::ivec3 shift_amount = glm::ivec3(0);
	if (dimension & x) shift_amount.x = to_ivec3(source.get_size(), 1).x / 2;
	if (dimension & y) shift_amount.y = to_ivec3(source.get_size(), 1).y / 2;
	if (dimension & z) shift_amount.z = to_ivec3(source.get_size(), 1).z / 2;
	return shift(source, shift_amount);
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

template<typename T>
inline void FFFT2::i_shift(T& source, T& target, fft_dimension dimension)
{
	glm::ivec3 shift_amount = glm::ivec3(0);
	if (dimension & x) shift_amount.x = to_ivec3(source.get_size(), 1).x / 2;
	if (dimension & y) shift_amount.y = to_ivec3(source.get_size(), 1).y / 2;
	if (dimension & z) shift_amount.z = to_ivec3(source.get_size(), 1).z / 2;
	shift(source, target, -shift_amount);
}

template<typename T>
inline std::shared_ptr<T> FFFT2::i_shift(T& source, fft_dimension dimension)
{
	glm::ivec3 shift_amount = glm::ivec3(0);
	if (dimension & x) shift_amount.x = to_ivec3(source.get_size(), 1).x / 2;
	if (dimension & y) shift_amount.y = to_ivec3(source.get_size(), 1).y / 2;
	if (dimension & z) shift_amount.z = to_ivec3(source.get_size(), 1).z / 2;
	return shift(source, -shift_amount);

}

template<typename T> 
void FFFT2::mixed_fft(T& source, T& target, fft_dimension dimension, size_t max_radix, bool inverse) {

	if (dimension != x && dimension != y && dimension != z) {
		std::cout << "[FFFT Error] FFFT::mixed_fft() is called with invalid dimension" << std::endl;
		ASSERT(false);
	}

	size_t array_size =
		dimension == x ? to_ivec3(source.get_size()).x :
		dimension == y ? to_ivec3(source.get_size()).y :
		dimension == z ? to_ivec3(source.get_size()).z : to_ivec3(source.get_size()).x;

	max_radix = std::max(max_radix, (size_t)1);

	fft_plan plan = create_plan(array_size, max_radix);

	T* source_p = &source;
	T* target_p = &target;

	for (int32_t i = 0; i < plan.iterations.size(); i++) {
		fft_iteration& iteration = plan.iterations[i];

		if (iteration.radix == fft_iteration::radix_dft)
			break;

		glm::ivec3 split_count = glm::ivec3(1, 1, 1);
		if (dimension == x) split_count.x = iteration.radix;
		if (dimension == y) split_count.y = iteration.radix;
		if (dimension == z) split_count.z = iteration.radix;

		glm::ivec3 group_count = glm::ivec3(1, 1, 1);
		if (dimension == x) group_count.x = array_size / iteration.chunk_size / iteration.radix;
		if (dimension == y) group_count.y = array_size / iteration.chunk_size / iteration.radix;
		if (dimension == z) group_count.z = array_size / iteration.chunk_size / iteration.radix;

		//std::cout << "radix:\t\t" << iteration.radix << ", " << group_count << ", " << iteration.chunk_size << std::endl;
		split(*source_p, *target_p, split_count, group_count);
		std::swap(source_p, target_p);
	}

	//std::cout << "----------------------" << std::endl;

	for (int32_t i = plan.iterations.size() - 1; i >= 0; i--) {
		fft_iteration& iteration = plan.iterations[i];

		glm::ivec3 group_count = glm::ivec3(1, 1, 1);
		if (dimension == x) group_count.x = array_size / iteration.chunk_size / iteration.radix;
		if (dimension == y) group_count.y = array_size / iteration.chunk_size / iteration.radix;
		if (dimension == z) group_count.z = array_size / iteration.chunk_size / iteration.radix;

		//std::cout << "radix:\t\t" << iteration.radix << ", " <<  group_count << ", " << iteration.chunk_size << std::endl;
		if (iteration.radix == fft_iteration::radix_dft) {
			dft(*source_p, *target_p, dimension, inverse, group_count);
			std::swap(source_p, target_p);
		}
		else {
			step(*source_p, *target_p, iteration.radix, dimension, inverse, group_count);
			std::swap(source_p, target_p);
		}
	}

	if (target_p != &source)
		copy(source, target, real_complex);

	divide(target, glm::vec2(glm::sqrt(array_size)));
}

template<typename T>
inline void FFFT2::fft(T& source, T& target, fft_dimension dimension, size_t max_radix)
{
	T* source_p = &source;
	T* target_p = &target;

	if (dimension & x) { mixed_fft(*source_p, *target_p, x, max_radix, false); std::swap(source_p, target_p); }
	if (dimension & y) { mixed_fft(*source_p, *target_p, y, max_radix, false); std::swap(source_p, target_p); }
	if (dimension & z) { mixed_fft(*source_p, *target_p, z, max_radix, false); std::swap(source_p, target_p); }

	if (target_p != &source)
		copy(source, target, real_complex);
}

template<typename T>
inline void FFFT2::i_fft(T& source, T& target, fft_dimension dimension, size_t max_radix)
{
	T* source_p = &source;
	T* target_p = &target;

	if (dimension & x) { mixed_fft(*source_p, *target_p, x, max_radix, true); std::swap(source_p, target_p); };
	if (dimension & y) { mixed_fft(*source_p, *target_p, y, max_radix, true); std::swap(source_p, target_p); };
	if (dimension & z) { mixed_fft(*source_p, *target_p, z, max_radix, true); std::swap(source_p, target_p); };

	if (target_p != &source)
		copy(source, target, real_complex);
}
