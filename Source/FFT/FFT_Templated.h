#pragma once
#include "FFT.h"

template<> inline constexpr FFFT::fft_dimension FFFT::default_fft_dimension<Buffer>()			{ return x; }
template<> inline constexpr FFFT::fft_dimension FFFT::default_fft_dimension<Texture1D>()		{ return x; }
template<> inline constexpr FFFT::fft_dimension FFFT::default_fft_dimension<Texture2D>()		{ return xy; }
template<> inline constexpr FFFT::fft_dimension FFFT::default_fft_dimension<Texture2DArray>()	{ return xyz; }
template<> inline constexpr FFFT::fft_dimension FFFT::default_fft_dimension<Texture3D>()		{ return xyz; }

template<typename T>
inline bool FFFT::is_complex(T& source)
{
	return source.ColorTextureFormat_channels() == 2 && (source.ColorTextureFormat_to_Type() == Texture3D::Type::HALF_FLOAT || source.ColorTextureFormat_to_Type() == Texture3D::Type::FLOAT);;
}

template<typename T>
inline bool FFFT::is_real(T& source)
{
	return source.ColorTextureFormat_channels() == 1 && (source.ColorTextureFormat_to_Type() == Texture3D::Type::HALF_FLOAT || source.ColorTextureFormat_to_Type() == Texture3D::Type::FLOAT);;
}

template<typename T>
inline void FFFT::fft(T& source, T& target, fft_dimension dimension, fft_algorithm algorithm, glm::ivec3 offset, glm::ivec3 size)
{
	dft(source, target, dimension, offset, size);
}

template<typename T>
inline void FFFT::dft(T& source, T& target, fft_dimension dimension, glm::ivec3 offset, glm::ivec3 size)
{
		
}
