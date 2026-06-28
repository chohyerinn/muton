package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.GridLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.example.myapplication.databinding.ActivityCalendarBinding
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale
import kotlin.math.abs

class CalendarActivity : BaseActivity() {

    private lateinit var binding: ActivityCalendarBinding
    private var isMonthMode = false
    private var selectedDateKey = ConversationRecordStore.getTodayDateKey()
    private var selectedCalendar: Calendar = Calendar.getInstance()
    private var visibleWeekStartCalendar: Calendar = startOfWeek(selectedCalendar)
    private var weekSwipeStartX = 0f
    private var weekSwipeStartY = 0f
    private var monthSwipeStartX = 0f
    private var monthSwipeStartY = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCalendarBinding.inflate(layoutInflater)
        setContentView(binding.root)

        applyIntentSelection(intent)

        binding.btnCalendarToggle.setOnClickListener {
            isMonthMode = !isMonthMode
            renderCalendarMode()
        }
        binding.btnMoreMenu.setOnClickListener {
            showMoreMenu()
        }
        bindWeekSwipeGesture()
        bindMonthSwipeGesture()

        renderCalendarMode()
        renderRecords()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        applyIntentSelection(intent)
        renderCalendarMode()
        renderRecords()
    }

    override fun onResume() {
        super.onResume()
        ConversationRecordStore.syncFromFirebase(this) {
            runOnUiThread {
                renderCalendarMode()
                renderRecords()
            }
        }
    }

    private fun renderCalendarMode() {
        updateMonthTitle()
        renderTopActionSelection(isMoreSelected = false)

        if (isMonthMode) {
            renderMonthGrid()
            animateCalendarSwitch(binding.weekStrip, false)
            animateCalendarSwitch(binding.monthHeaderStrip, true)
            animateCalendarSwitch(binding.monthGrid, true)
            binding.dragHandle.layoutParams =
                (binding.dragHandle.layoutParams as androidx.constraintlayout.widget.ConstraintLayout.LayoutParams)
                    .apply { topToBottom = binding.monthGrid.id }
        } else {
            renderWeekStrip()
            animateCalendarSwitch(binding.monthGrid, false)
            animateCalendarSwitch(binding.monthHeaderStrip, false)
            animateCalendarSwitch(binding.weekStrip, true)
            binding.dragHandle.layoutParams =
                (binding.dragHandle.layoutParams as androidx.constraintlayout.widget.ConstraintLayout.LayoutParams)
                    .apply { topToBottom = binding.weekStrip.id }
        }

        binding.dragHandle.requestLayout()
    }

    private fun renderWeekStrip() {
        val dayOfWeekFormatter = SimpleDateFormat("EEEEE", Locale.ENGLISH)
        val dayFormatter = SimpleDateFormat("d", Locale.ENGLISH)
        val dateKeyFormatter = SimpleDateFormat("yyyy-MM-dd", Locale.KOREA)

        binding.weekStrip.removeAllViews()
        getVisibleWeekDates().forEach { date ->
            val itemView = LayoutInflater.from(this)
                .inflate(R.layout.item_calendar_day, binding.weekStrip, false)
            val dayName = itemView.findViewById<TextView>(R.id.dayName)
            val dayNumber = itemView.findViewById<TextView>(R.id.dayNumber)
            val dots = itemView.findViewById<LinearLayout>(R.id.weekRecordDots)

            dayName.text = dayOfWeekFormatter.format(date.time)
            dayNumber.text = dayFormatter.format(date.time)

            val dateKey = dateKeyFormatter.format(date.time)
            if (dateKey == selectedDateKey) {
                dayNumber.setBackgroundResource(R.drawable.bg_calendar_selected_day)
                dayNumber.setTextColor(getColor(R.color.muton_text))
            } else {
                dayNumber.background = null
                dayNumber.setTextColor(getColor(R.color.muton_muted))
            }

            dayName.setTextColor(getColor(R.color.muton_calendar_weekday))

            val recordCount = ConversationRecordStore.getRecordCountForDate(this, dateKey).coerceAtMost(3)
            repeat(recordCount) {
                dots.addView(
                    View(this).apply {
                        background = ContextCompat.getDrawable(this@CalendarActivity, R.drawable.bg_record_dot)
                        layoutParams = LinearLayout.LayoutParams(4.dp(), 4.dp()).apply {
                            marginStart = 1.dp()
                            marginEnd = 1.dp()
                        }
                    },
                )
            }

            itemView.setOnClickListener {
                selectedDateKey = dateKey
                selectedCalendar = (date.clone() as Calendar)
                visibleWeekStartCalendar = startOfWeek(selectedCalendar)
                renderCalendarMode()
                renderRecords()
            }
            itemView.setOnTouchListener(createWeekSwipeTouchListener())

            binding.weekStrip.addView(itemView)
        }
    }

    private fun renderMonthGrid() {
        val dateKeyFormatter = SimpleDateFormat("yyyy-MM-dd", Locale.KOREA)
        val dayFormatter = SimpleDateFormat("d", Locale.ENGLISH)
        val monthStart = (selectedCalendar.clone() as Calendar).apply {
            set(Calendar.DAY_OF_MONTH, 1)
        }
        val activeMonth = monthStart.get(Calendar.MONTH)
        val firstDayOfWeek = monthStart.get(Calendar.DAY_OF_WEEK)
        val cells = ((firstDayOfWeek - Calendar.SUNDAY) +
            monthStart.getActualMaximum(Calendar.DAY_OF_MONTH) + 6) / 7 * 7
        val cellWidth = resources.displayMetrics.widthPixels - (28 * resources.displayMetrics.density).toInt()

        binding.monthGrid.removeAllViews()

        repeat(cells) { index ->
            val cellDate = (monthStart.clone() as Calendar).apply {
                add(Calendar.DAY_OF_MONTH, index - (firstDayOfWeek - Calendar.SUNDAY))
            }
            val itemView = LayoutInflater.from(this)
                .inflate(R.layout.item_calendar_month_day, binding.monthGrid, false)
            val params = GridLayout.LayoutParams().apply {
                width = cellWidth / 7
                height = ViewGroup.LayoutParams.WRAP_CONTENT
            }
            itemView.layoutParams = params

            val dayNumber = itemView.findViewById<TextView>(R.id.monthDayNumber)
            val dots = itemView.findViewById<LinearLayout>(R.id.monthRecordDots)
            val dateKey = dateKeyFormatter.format(cellDate.time)
            val isActiveMonth = cellDate.get(Calendar.MONTH) == activeMonth

            dayNumber.text = dayFormatter.format(cellDate.time)
            dayNumber.alpha = if (isActiveMonth) 1f else 0.24f

            if (dateKey == selectedDateKey && isActiveMonth) {
                dayNumber.setBackgroundResource(R.drawable.bg_calendar_selected_month_day)
                dayNumber.setTextColor(getColor(android.R.color.white))
            } else {
                dayNumber.background = null
                dayNumber.setTextColor(
                    getColor(
                        if (isActiveMonth) R.color.muton_text else R.color.muton_muted,
                    ),
                )
            }

            val recordCount = if (isActiveMonth) {
                ConversationRecordStore.getRecordCountForDate(this, dateKey).coerceAtMost(3)
            } else {
                0
            }
            repeat(recordCount) {
                dots.addView(
                    View(this).apply {
                        background = ContextCompat.getDrawable(this@CalendarActivity, R.drawable.bg_record_dot)
                        layoutParams = LinearLayout.LayoutParams(4.dp(), 4.dp()).apply {
                            marginStart = 1.dp()
                            marginEnd = 1.dp()
                        }
                    },
                )
            }

            itemView.setOnClickListener {
                selectedDateKey = dateKey
                selectedCalendar = (cellDate.clone() as Calendar)
                visibleWeekStartCalendar = startOfWeek(selectedCalendar)
                renderCalendarMode()
                renderRecords()
            }
            itemView.setOnTouchListener(createMonthSwipeTouchListener())

            binding.monthGrid.addView(itemView)
        }
    }

    private fun renderRecords() {
        val records = ConversationRecordStore.getRecordsForDate(
            context = this,
            dateKey = selectedDateKey,
        )
        val selectedLabelFormatter = SimpleDateFormat("EEEE d", Locale.ENGLISH)
        binding.selectedDayLabel.text = selectedLabelFormatter.format(selectedCalendar.time)

        binding.recordCount.text = getString(R.string.calendar_records, records.size)
        binding.recordContainer.removeAllViews()

        if (records.isEmpty()) {
            val emptyView = TextView(this).apply {
                text = getString(R.string.calendar_empty)
                textSize = 15f
                setTextColor(getColor(R.color.muton_muted))
            }
            binding.recordContainer.addView(emptyView)
            binding.recordScroll.alpha = 0f
            binding.recordScroll.translationY = 12.dp().toFloat()
            binding.recordScroll.animate()
                .alpha(1f)
                .translationY(0f)
                .setDuration(180L)
                .start()
            return
        }

        records.reversed().forEachIndexed { index, record ->
            val card = LayoutInflater.from(this)
                .inflate(R.layout.item_calendar_record, binding.recordContainer, false)
            card.findViewById<TextView>(R.id.recordTitle).text = record.title
            card.findViewById<TextView>(R.id.recordTime).text = record.timeRange
            val star = card.findViewById<ImageView>(R.id.recordStar)
            star.setColorFilter(
                getColor(
                    if (record.isFavorite) R.color.muton_gold else android.R.color.white,
                ),
            )
            star.alpha = if (record.isFavorite) 1f else 0.72f
            star.setOnClickListener {
                ConversationRecordStore.toggleFavorite(this, record.dateKey, record.createdAt)
                renderRecords()
            }
            card.setOnClickListener {
                startActivity(
                    Intent(this, RecordDetailActivity::class.java).apply {
                        putExtra(RecordDetailActivity.EXTRA_DATE_KEY, record.dateKey)
                        putExtra(RecordDetailActivity.EXTRA_CREATED_AT, record.createdAt)
                    },
                )
            }
            binding.recordContainer.addView(card)
            card.alpha = 0f
            card.translationY = 10.dp().toFloat()
            card.animate()
                .alpha(1f)
                .translationY(0f)
                .setStartDelay((index * 32L).coerceAtMost(120L))
                .setDuration(180L)
                .start()
        }
    }

    private fun updateMonthTitle() {
        val formatter = SimpleDateFormat("MMMM yyyy", Locale.ENGLISH)
        binding.monthTitle.text = formatter.format(selectedCalendar.time)
    }

    private fun animateCalendarSwitch(target: View, show: Boolean) {
        if (show) {
            target.visibility = View.VISIBLE
            target.alpha = 0f
            target.translationY = 10.dp().toFloat()
            target.animate()
                .alpha(1f)
                .translationY(0f)
                .setDuration(180L)
                .start()
        } else if (target.visibility == View.VISIBLE) {
            target.animate()
                .alpha(0f)
                .translationY(-6.dp().toFloat())
                .setDuration(140L)
                .withEndAction {
                    target.visibility = View.GONE
                    target.alpha = 1f
                    target.translationY = 0f
                }
                .start()
        } else {
            target.visibility = View.GONE
        }
    }

    private fun showMoreMenu() {
        renderTopActionSelection(isMoreSelected = true)
        val popupView = LayoutInflater.from(this).inflate(R.layout.popup_calendar_menu, null, false)
        val popupWindow = PopupWindow(
            popupView,
            (140 * resources.displayMetrics.density).toInt(),
            ViewGroup.LayoutParams.WRAP_CONTENT,
            true,
        )

        popupWindow.setBackgroundDrawable(
            ContextCompat.getDrawable(this, android.R.color.transparent),
        )
        popupWindow.elevation = 0f
        popupWindow.isOutsideTouchable = true
        popupWindow.setOnDismissListener {
            renderTopActionSelection(isMoreSelected = false)
        }

        popupView.findViewById<View>(R.id.menuTodayRow).setOnClickListener {
            popupWindow.dismiss()
            selectedDateKey = ConversationRecordStore.getTodayDateKey()
            selectedCalendar = Calendar.getInstance()
            renderCalendarMode()
            renderRecords()
        }

        popupView.findViewById<View>(R.id.menuHomeRow).setOnClickListener {
            popupWindow.dismiss()
            startActivity(Intent(this, HomeActivity::class.java))
        }

        popupWindow.showAsDropDown(binding.btnMoreMenu, -88.dp(), 14.dp())
    }

    private fun renderTopActionSelection(isMoreSelected: Boolean) {
        binding.btnCalendarToggle.background = ContextCompat.getDrawable(
            this,
            if (isMoreSelected) android.R.color.transparent else R.drawable.bg_calendar_more_circle,
        )
        binding.btnMoreMenu.background = ContextCompat.getDrawable(
            this,
            if (isMoreSelected) R.drawable.bg_calendar_more_circle else android.R.color.transparent,
        )
    }

    private fun bindWeekSwipeGesture() {
        val listener = createWeekSwipeTouchListener()
        binding.weekStrip.setOnTouchListener(listener)
        binding.dragHandle.setOnTouchListener(listener)
    }

    private fun bindMonthSwipeGesture() {
        val listener = createMonthSwipeTouchListener()
        binding.monthHeaderStrip.setOnTouchListener(listener)
        binding.monthGrid.setOnTouchListener(listener)
    }

    private fun createWeekSwipeTouchListener(): View.OnTouchListener {
        return View.OnTouchListener { view, event ->
            if (isMonthMode) return@OnTouchListener false

            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    weekSwipeStartX = event.rawX
                    weekSwipeStartY = event.rawY
                    true
                }

                MotionEvent.ACTION_UP -> {
                    val deltaX = event.rawX - weekSwipeStartX
                    val deltaY = event.rawY - weekSwipeStartY
                    val minDistance = 72.dp().toFloat()

                    val handled =
                        when {
                            deltaY > minDistance && abs(deltaY) > abs(deltaX) -> {
                                switchToMonthMode()
                                true
                            }

                            deltaX < -minDistance && abs(deltaX) > abs(deltaY) -> {
                                moveWeekBy(1)
                                true
                            }

                            deltaX > minDistance && abs(deltaX) > abs(deltaY) -> {
                                moveWeekBy(-1)
                                true
                            }

                            else -> false
                        }

                    if (!handled && view.id != R.id.weekStrip && view.id != R.id.dragHandle) {
                        view.performClick()
                    }

                    handled
                }

                MotionEvent.ACTION_CANCEL -> false

                else -> true
            }
        }
    }

    private fun createMonthSwipeTouchListener(): View.OnTouchListener {
        return View.OnTouchListener { view, event ->
            if (!isMonthMode) return@OnTouchListener false

            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    monthSwipeStartX = event.rawX
                    monthSwipeStartY = event.rawY
                    true
                }

                MotionEvent.ACTION_UP -> {
                    val deltaX = event.rawX - monthSwipeStartX
                    val deltaY = event.rawY - monthSwipeStartY
                    val minDistance = 72.dp().toFloat()

                    val handled = when {
                        deltaY < -minDistance && abs(deltaY) > abs(deltaX) -> {
                            switchToWeekMode()
                            true
                        }

                        deltaX < -minDistance && abs(deltaX) > abs(deltaY) -> {
                            moveMonthBy(1)
                            true
                        }

                        deltaX > minDistance && abs(deltaX) > abs(deltaY) -> {
                            moveMonthBy(-1)
                            true
                        }

                        else -> false
                    }

                    if (!handled && view.id != R.id.monthGrid && view.id != R.id.monthHeaderStrip) {
                        view.performClick()
                    }

                    handled
                }

                MotionEvent.ACTION_CANCEL -> false

                else -> true
            }
        }
    }

    private fun moveWeekBy(weekOffset: Int) {
        visibleWeekStartCalendar = (visibleWeekStartCalendar.clone() as Calendar).apply {
            add(Calendar.DAY_OF_MONTH, weekOffset * 7)
        }
        selectedCalendar = (selectedCalendar.clone() as Calendar).apply {
            add(Calendar.DAY_OF_MONTH, weekOffset * 7)
        }
        selectedDateKey = formatDateKey(selectedCalendar)
        renderCalendarMode()
        renderRecords()
    }

    private fun switchToMonthMode() {
        if (isMonthMode) return
        isMonthMode = true
        renderCalendarMode()
    }

    private fun switchToWeekMode() {
        if (!isMonthMode) return
        isMonthMode = false
        visibleWeekStartCalendar = startOfWeek(selectedCalendar)
        renderCalendarMode()
    }

    private fun moveMonthBy(monthOffset: Int) {
        selectedCalendar = (selectedCalendar.clone() as Calendar).apply {
            add(Calendar.MONTH, monthOffset)
        }
        selectedDateKey = formatDateKey(selectedCalendar)
        visibleWeekStartCalendar = startOfWeek(selectedCalendar)
        renderCalendarMode()
        renderRecords()
    }

    private fun getVisibleWeekDates(): List<Calendar> {
        return List(7) { index ->
            (visibleWeekStartCalendar.clone() as Calendar).apply {
                add(Calendar.DAY_OF_MONTH, index)
            }
        }
    }

    private fun applyIntentSelection(intent: Intent?) {
        val targetDateKey = intent?.getStringExtra(EXTRA_SELECTED_DATE_KEY).orEmpty()
        if (targetDateKey.isBlank()) return

        selectedDateKey = targetDateKey
        selectedCalendar = calendarFromDateKey(targetDateKey)
        visibleWeekStartCalendar = startOfWeek(selectedCalendar)
    }

    private fun calendarFromDateKey(dateKey: String): Calendar {
        return Calendar.getInstance().apply {
            val parts = dateKey.split("-")
            if (parts.size == 3) {
                set(Calendar.YEAR, parts[0].toInt())
                set(Calendar.MONTH, parts[1].toInt() - 1)
                set(Calendar.DAY_OF_MONTH, parts[2].toInt())
            }
        }
    }

    private fun startOfWeek(source: Calendar): Calendar {
        return (source.clone() as Calendar).apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
            val offset = get(Calendar.DAY_OF_WEEK) - Calendar.SUNDAY
            add(Calendar.DAY_OF_MONTH, -offset)
        }
    }

    private fun formatDateKey(calendar: Calendar): String {
        return SimpleDateFormat("yyyy-MM-dd", Locale.KOREA).format(calendar.time)
    }

    private fun Int.dp(): Int = (this * resources.displayMetrics.density).toInt()

    companion object {
        const val EXTRA_SELECTED_DATE_KEY = "extra_selected_date_key"
    }
}
