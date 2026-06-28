package com.example.myapplication

import android.content.Context
import androidx.appcompat.app.AppCompatDelegate

object DarkModeManager {

    private const val PREFS_NAME = "muton_dark_mode"
    private const val KEY_ENABLED = "enabled"

    fun apply(context: Context) {
        AppCompatDelegate.setDefaultNightMode(
            if (isEnabled(context)) {
                AppCompatDelegate.MODE_NIGHT_YES
            } else {
                AppCompatDelegate.MODE_NIGHT_NO
            },
        )
    }

    fun isEnabled(context: Context): Boolean {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(KEY_ENABLED, false)
    }

    fun setEnabled(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_ENABLED, enabled)
            .apply()
    }
}
