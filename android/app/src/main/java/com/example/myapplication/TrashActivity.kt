package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.widget.TextView
import com.example.myapplication.databinding.ActivityTrashBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class TrashActivity : BaseActivity() {

    private lateinit var binding: ActivityTrashBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTrashBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.backButton.setOnClickListener { finish() }
        renderTrash()
    }

    override fun onResume() {
        super.onResume()
        ConversationRecordStore.syncFromFirebase(this) {
            runOnUiThread {
                renderTrash()
            }
        }
    }

    private fun renderTrash() {
        val records = ConversationRecordStore.getTrashedRecords(this)
        binding.trashContainer.removeAllViews()

        if (records.isEmpty()) {
            binding.trashContainer.addView(
                TextView(this).apply {
                    text = getString(R.string.trash_empty)
                    textSize = 14f
                    setTextColor(getColor(R.color.muton_muted))
                },
            )
            return
        }

        records.forEach { record ->
            val row = LayoutInflater.from(this)
                .inflate(R.layout.item_trash_record, binding.trashContainer, false)
            row.findViewById<TextView>(R.id.txtTrashItemTitle).text = record.title
            row.findViewById<TextView>(R.id.txtTrashItemTime).text = formatRecordDate(record.createdAt)
            row.setOnClickListener {
                startActivity(
                    Intent(this, RecordDetailActivity::class.java).apply {
                        putExtra(RecordDetailActivity.EXTRA_DATE_KEY, record.dateKey)
                        putExtra(RecordDetailActivity.EXTRA_CREATED_AT, record.createdAt)
                        putExtra(RecordDetailActivity.EXTRA_FROM_TRASH, true)
                    },
                )
            }
            row.findViewById<android.widget.ImageView>(R.id.btnPermanentDelete).setOnClickListener {
                ConversationRecordStore.permanentlyDelete(this, record.dateKey, record.createdAt)
                renderTrash()
            }
            binding.trashContainer.addView(row)
        }
    }

    private fun formatRecordDate(createdAt: Long): String {
        return SimpleDateFormat("yy년 M월 d일", Locale.KOREA).format(Date(createdAt))
    }
}
