package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import com.example.myapplication.databinding.ActivityRecordDetailBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class RecordDetailActivity : BaseActivity() {

    private lateinit var binding: ActivityRecordDetailBinding
    private var dateKey: String = ""
    private var createdAt: Long = 0L
    private var fromTrash: Boolean = false
    private var returnHome: Boolean = false
    private val displayDateFormatter = SimpleDateFormat("yy년 M월 d일", Locale.KOREA)
    private val displayTimeFormatter = SimpleDateFormat("HH:mm", Locale.KOREA)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityRecordDetailBinding.inflate(layoutInflater)
        setContentView(binding.root)

        dateKey = intent.getStringExtra(EXTRA_DATE_KEY).orEmpty()
        createdAt = intent.getLongExtra(EXTRA_CREATED_AT, 0L)
        fromTrash = intent.getBooleanExtra(EXTRA_FROM_TRASH, false)
        returnHome = intent.getBooleanExtra(EXTRA_RETURN_HOME, false)

        binding.backButton.setOnClickListener { finish() }
        binding.btnMoveToTrash.setOnClickListener {
            if (fromTrash) {
                ConversationRecordStore.restoreFromTrash(this, dateKey, createdAt) {
                    runOnUiThread {
                        finish()
                    }
                }
            } else {
                ConversationRecordStore.moveToTrash(this, dateKey, createdAt)
                if (returnHome) {
                    startActivity(
                        Intent(this, HomeActivity::class.java).apply {
                            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                        },
                    )
                } else {
                    startActivity(
                        Intent(this, CalendarActivity::class.java).apply {
                            putExtra(CalendarActivity.EXTRA_SELECTED_DATE_KEY, dateKey)
                            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                        },
                    )
                }
                finish()
            }
        }

        renderRecord()
    }

    private fun renderRecord() {
        val record = ConversationRecordStore.findRecord(this, dateKey, createdAt) ?: run {
            finish()
            return
        }

        binding.txtDetailDate.text = displayDateFormatter.format(Date(record.startedAt))
        binding.txtDetailTime.text = displayTimeFormatter.format(Date(record.startedAt))
        binding.txtDetailSummaryCard.text = record.title
        binding.detailConversationWrap.removeAllViews()
        binding.btnMoveToTrash.setImageResource(
            if (fromTrash) android.R.drawable.ic_menu_revert else android.R.drawable.ic_menu_delete,
        )

        addBubbles(record.selfSpeech, record.selfSummary, true)
        addBubbles(record.otherSpeech, record.otherSummary, false)
        binding.detailScroll.post {
            binding.detailScroll.scrollTo(0, 0)
        }
    }

    private fun addBubbles(speech: String, summary: String, isSelf: Boolean) {
        val speechParts = speech.lines().map { it.trim() }.filter { it.isNotBlank() }
        val summaryParts = summary.lines().map { it.trim() }.filter { it.isNotBlank() }
        val bubbleCount = maxOf(speechParts.size, summaryParts.size)

        if (bubbleCount == 0) return

        repeat(bubbleCount) { index ->
            val speechText = speechParts.getOrNull(index).orEmpty()
            val summaryText = summaryParts.getOrNull(index).orEmpty()
            if (speechText.isBlank() && summaryText.isBlank()) return@repeat

            val bubble = LayoutInflater.from(this)
                .inflate(R.layout.item_record_detail_bubble, binding.detailConversationWrap, false) as LinearLayout
            bubble.background = getDrawable(
                if (isSelf) R.drawable.bg_camera_bubble_light else R.drawable.bg_camera_bubble_dark,
            )

            val speechView = bubble.findViewById<TextView>(R.id.txtBubbleSpeech)
            val summaryView = bubble.findViewById<TextView>(R.id.txtBubbleSummary)

            speechView.text = speechText
            summaryView.text = summaryText
            summaryView.visibility = if (summaryText.isBlank()) View.GONE else View.VISIBLE

            binding.detailConversationWrap.addView(bubble)
        }
    }

    companion object {
        const val EXTRA_DATE_KEY = "extra_date_key"
        const val EXTRA_CREATED_AT = "extra_created_at"
        const val EXTRA_FROM_TRASH = "extra_from_trash"
        const val EXTRA_RETURN_HOME = "extra_return_home"
    }
}
