package com.example.myapplication

import android.content.Context
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

abstract class BaseActivity : AppCompatActivity() {

    private var appliedTextSizeOption: AppTextScaleManager.TextSizeOption? = null
    private var appliedDarkMode: Boolean? = null

    override fun attachBaseContext(newBase: Context) {
        super.attachBaseContext(AppTextScaleManager.wrapContext(newBase))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        DarkModeManager.apply(this)
        appliedTextSizeOption = AppTextScaleManager.getTextSizeOption(this)
        appliedDarkMode = DarkModeManager.isEnabled(this)
        super.onCreate(savedInstanceState)
    }

    override fun onResume() {
        super.onResume()
        val currentOption = AppTextScaleManager.getTextSizeOption(this)
        val currentDarkMode = DarkModeManager.isEnabled(this)
        if (appliedTextSizeOption != null && appliedTextSizeOption != currentOption) {
            appliedTextSizeOption = currentOption
            recreate()
        } else if (appliedDarkMode != null && appliedDarkMode != currentDarkMode) {
            appliedDarkMode = currentDarkMode
            DarkModeManager.apply(this)
            recreate()
        }
    }
}
