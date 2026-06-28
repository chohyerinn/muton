package com.example.myapplication

import android.animation.ValueAnimator
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.widget.ImageView
import android.widget.PopupWindow
import android.widget.TextView
import androidx.core.animation.doOnEnd
import androidx.core.content.ContextCompat
import androidx.core.view.doOnLayout
import com.example.myapplication.databinding.ActivityHomeBinding
import kotlin.math.max
import kotlin.math.min

class HomeActivity : BaseActivity() {

    private lateinit var binding: ActivityHomeBinding
    private var maxSlideDistance = 0f
    private var sliderStartWidth = 0
    private var downRawX = 0f
    private var dragStartOffset = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHomeBinding.inflate(layoutInflater)
        setContentView(binding.root)

        renderGreeting()

        binding.btnMenu.setOnClickListener {
            showMenu()
        }

        renderFavorites()
        binding.btnJournal.setOnClickListener {
            startActivity(Intent(this, CalendarActivity::class.java))
        }

        binding.sliderTrack.doOnLayout {
            sliderStartWidth = binding.sliderThumb.width + binding.sliderThumb.marginStartPx() + 12.dp()
            maxSlideDistance =
                (binding.sliderTrack.width
                    - binding.sliderThumb.width
                    - binding.sliderThumb.marginStartPx()
                    - binding.sliderThumb.marginEndPx()).toFloat()
            updateSliderUi(0f)
        }

        binding.sliderThumb.isClickable = false
        binding.sliderThumb.isFocusable = false
        val sliderTouchListener = View.OnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    downRawX = event.rawX
                    dragStartOffset = binding.sliderThumb.translationX
                    true
                }

                MotionEvent.ACTION_MOVE -> {
                    val delta = event.rawX - downRawX
                    val nextOffset = min(
                        max(dragStartOffset + delta, 0f),
                        maxSlideDistance,
                    )
                    updateSliderUi(nextOffset)
                    true
                }

                MotionEvent.ACTION_UP,
                MotionEvent.ACTION_CANCEL,
                -> {
                    val travelled = binding.sliderThumb.translationX
                    if (travelled > maxSlideDistance * 0.78f) {
                        animateSliderTo(maxSlideDistance, 120L) {
                            startActivity(Intent(this, MainActivity::class.java))
                            updateSliderUi(0f)
                        }
                    } else {
                        animateSliderTo(0f, 160L)
                    }
                    true
                }

                else -> false
            }
        }
        binding.sliderTrack.setOnTouchListener(sliderTouchListener)
        binding.sliderThumb.setOnTouchListener(sliderTouchListener)
    }

    override fun onResume() {
        super.onResume()
        updateSliderUi(0f)
        renderGreeting()
        ConversationRecordStore.syncFromFirebase(this) {
            runOnUiThread {
                renderFavorites()
            }
        }
    }

    private fun renderGreeting() {
        FirebaseUserStore.loadProfile(this) { profile ->
            runOnUiThread {
                binding.txtGreeting.text = "hello, ${profile.displayName}!"
                binding.imgLogoAvatar.setImageDrawable(null)
                binding.imgLogoAvatar.visibility = View.GONE
                binding.logoMark.visibility = View.VISIBLE
            }
        }
    }

    private fun showMenu() {
        val popupView = LayoutInflater.from(this).inflate(R.layout.popup_home_menu, null, false)
        val popupWindow = PopupWindow(
            popupView,
            (140 * resources.displayMetrics.density).toInt(),
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT,
            true,
        )

        popupWindow.setBackgroundDrawable(
            ContextCompat.getDrawable(this, android.R.color.transparent),
        )
        popupWindow.elevation = 0f
        popupWindow.isOutsideTouchable = true

        popupView.findViewById<android.view.View>(R.id.menuSettingRow).setOnClickListener {
            popupWindow.dismiss()
            startActivity(Intent(this@HomeActivity, SettingsActivity::class.java))
        }

        popupView.findViewById<android.view.View>(R.id.menuTrashRow).setOnClickListener {
            popupWindow.dismiss()
            startActivity(Intent(this@HomeActivity, TrashActivity::class.java))
        }

        popupWindow.showAsDropDown(binding.btnMenu, -116.dp(), 8.dp())
    }

    private fun renderFavorites() {
        val favorites = ConversationRecordStore.getFavoriteRecords(this, null)
        binding.favoriteList.removeAllViews()

        if (favorites.isEmpty()) {
            (binding.emptyFavoriteText.parent as? android.view.ViewGroup)?.removeView(binding.emptyFavoriteText)
            binding.emptyFavoriteText.visibility = android.view.View.VISIBLE
            binding.favoriteList.addView(binding.emptyFavoriteText)
            return
        }

        binding.emptyFavoriteText.visibility = android.view.View.GONE
        favorites.forEach { record ->
            val row = LayoutInflater.from(this)
                .inflate(R.layout.item_home_favorite, binding.favoriteList, false)
            row.findViewById<TextView>(R.id.favoriteTitle).text = record.title
            row.findViewById<ImageView>(R.id.favoriteStar).apply {
                setOnClickListener {
                    ConversationRecordStore.toggleFavorite(this@HomeActivity, record.dateKey, record.createdAt)
                    renderFavorites()
                }
            }
            row.setOnClickListener {
                startActivity(
                    Intent(this, RecordDetailActivity::class.java).apply {
                        putExtra(RecordDetailActivity.EXTRA_DATE_KEY, record.dateKey)
                        putExtra(RecordDetailActivity.EXTRA_CREATED_AT, record.createdAt)
                        putExtra(RecordDetailActivity.EXTRA_RETURN_HOME, true)
                    },
                )
            }
            binding.favoriteList.addView(row)
        }
    }

    private fun android.view.View.marginStartPx(): Int {
        val params = layoutParams as? android.view.ViewGroup.MarginLayoutParams
        return params?.marginStart ?: 0
    }

    private fun android.view.View.marginEndPx(): Int {
        val params = layoutParams as? android.view.ViewGroup.MarginLayoutParams
        return params?.marginEnd ?: 0
    }

    private fun Int.dp(): Int = (this * resources.displayMetrics.density).toInt()

    private fun updateSliderUi(offset: Float) {
        binding.sliderThumb.translationX = offset
        updateSliderVisuals(offset)
    }

    private fun updateSliderVisuals(offset: Float) {
        val normalizedProgress =
            if (maxSlideDistance <= 0f) 0f else (offset / maxSlideDistance).coerceIn(0f, 1f)
        val progressWidth = sliderStartWidth + offset.toInt()
        binding.sliderProgress.layoutParams = binding.sliderProgress.layoutParams.apply {
            width = progressWidth
        }
        binding.sliderProgress.requestLayout()
        binding.txtSliderCameraOn.alpha = 0.62f - (normalizedProgress * 0.42f)
        binding.txtSliderCameraOn.translationX = (1f - normalizedProgress) * 6.dp()
    }

    private fun animateSliderTo(target: Float, duration: Long, onEnd: (() -> Unit)? = null) {
        ValueAnimator.ofFloat(binding.sliderThumb.translationX, target).apply {
            this.duration = duration
            addUpdateListener { animator ->
                updateSliderUi(animator.animatedValue as Float)
            }
            doOnEnd {
                onEnd?.invoke()
            }
            start()
        }
    }
}
