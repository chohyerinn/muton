package com.example.myapplication

import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.content.Intent
import android.os.Bundle
import android.view.animation.AccelerateDecelerateInterpolator
import com.example.myapplication.databinding.ActivitySplashBinding

class SplashActivity : BaseActivity() {

    private lateinit var binding: ActivitySplashBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySplashBinding.inflate(layoutInflater)
        setContentView(binding.root)

        playSplashAnimation()
    }

    private fun playSplashAnimation() {
        val logoRise = ObjectAnimator.ofFloat(binding.imgSplashLogo, "translationY", 28f, 0f).apply {
            duration = 720L
            startDelay = 220L
            interpolator = AccelerateDecelerateInterpolator()
        }
        val logoFade = ObjectAnimator.ofFloat(binding.imgSplashLogo, "alpha", 0f, 1f).apply {
            duration = 640L
            startDelay = 220L
            interpolator = AccelerateDecelerateInterpolator()
        }
        val logoScaleX = ObjectAnimator.ofFloat(binding.imgSplashLogo, "scaleX", 0.92f, 1f).apply {
            duration = 640L
            startDelay = 220L
            interpolator = AccelerateDecelerateInterpolator()
        }
        val logoScaleY = ObjectAnimator.ofFloat(binding.imgSplashLogo, "scaleY", 0.92f, 1f).apply {
            duration = 640L
            startDelay = 220L
            interpolator = AccelerateDecelerateInterpolator()
        }
        val muFade = ObjectAnimator.ofFloat(binding.txtSplashMuOverlay, "alpha", 0f, 1f).apply {
            duration = 420L
            startDelay = 420L
            interpolator = AccelerateDecelerateInterpolator()
        }
        val muRise = ObjectAnimator.ofFloat(binding.txtSplashMuOverlay, "translationY", 12f, 0f).apply {
            duration = 420L
            startDelay = 420L
            interpolator = AccelerateDecelerateInterpolator()
        }

        AnimatorSet().apply {
            playTogether(logoRise, logoFade, logoScaleX, logoScaleY, muFade, muRise)
            start()
        }

        binding.brandStage.postDelayed({
            startActivity(Intent(this, LoginActivity::class.java))
            finish()
        }, 1580L)
    }
}
