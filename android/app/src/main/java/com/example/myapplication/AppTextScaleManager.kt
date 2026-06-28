package com.example.myapplication

import android.content.Context
import android.content.res.Configuration

object AppTextScaleManager {

    private const val PREFS_NAME = "muton_display_settings"
    private const val KEY_TEXT_SIZE = "text_size_option"

    enum class TextSizeOption(val value: String, val fontScale: Float) {
        SMALL("small", 0.9f),
        MEDIUM("medium", 1.0f),
        LARGE("large", 1.15f),
        ;

        companion object {
            fun fromValue(value: String?): TextSizeOption {
                return entries.firstOrNull { it.value == value } ?: MEDIUM
            }
        }
    }

    fun getTextSizeOption(context: Context): TextSizeOption {
        val savedValue = context
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_TEXT_SIZE, TextSizeOption.MEDIUM.value)
        return TextSizeOption.fromValue(savedValue)
    }

    fun saveTextSizeOption(context: Context, option: TextSizeOption) {
        context
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_TEXT_SIZE, option.value)
            .apply()
    }

    fun wrapContext(context: Context): Context {
        val option = getTextSizeOption(context)
        val configuration = Configuration(context.resources.configuration).apply {
            fontScale = option.fontScale
        }
        return context.createConfigurationContext(configuration)
    }
}
