package com.example.myapplication

import android.content.Context
import android.util.Log
import com.google.firebase.FirebaseApp
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.DocumentSnapshot
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

data class ConversationRecord(
    val title: String,
    val subtitle: String,
    val dateKey: String,
    val timeRange: String,
    val startedAt: Long,
    val createdAt: Long,
    val isFavorite: Boolean,
    val selfSpeech: String = "",
    val selfSummary: String = "",
    val otherSpeech: String = "",
    val otherSummary: String = "",
    val isTrashed: Boolean = false,
)

object ConversationRecordStore {

    private const val PREFS_NAME = "muton_records"
    private const val TAG = "ConversationRecordStore"

    fun saveTodayRecord(
        context: Context,
        title: String,
        subtitle: String,
        startedAt: Long,
        endedAt: Long = System.currentTimeMillis(),
        selfSpeech: String = "",
        selfSummary: String = "",
        otherSpeech: String = "",
        otherSummary: String = "",
    ) {
        val dateKey = formatDateKey(endedAt)
        val records = loadRecords(context, dateKey).toMutableList()
        val record = ConversationRecord(
            title = title,
            subtitle = subtitle,
            dateKey = dateKey,
            timeRange = buildTimeRange(startedAt, endedAt),
            startedAt = startedAt,
            createdAt = endedAt,
            isFavorite = false,
            selfSpeech = selfSpeech,
            selfSummary = selfSummary,
            otherSpeech = otherSpeech,
            otherSummary = otherSummary,
            isTrashed = false,
        )
        records.add(record)
        persistRecords(context, dateKey, records)
        saveRecordToFirebase(context, record, startedAt, endedAt)
    }

    fun syncFromFirebase(
        context: Context,
        onComplete: (() -> Unit)? = null,
    ) {
        val uid = currentUserUid(context)
        if (uid.isNullOrBlank() || FirebaseApp.getApps(context).isEmpty()) {
            onComplete?.invoke()
            return
        }

        FirebaseFirestore.getInstance()
            .collection("users")
            .document(uid)
            .collection("records")
            .get()
            .addOnSuccessListener { snapshot ->
                val groupedByDate = snapshot.documents
                    .mapNotNull { document -> document.toConversationRecord() }
                    .groupBy { it.dateKey }

                val prefs = prefs(context)
                val editor = prefs.edit().clear()
                groupedByDate.forEach { (dateKey, records) ->
                    val jsonArray = JSONArray()
                    records.sortedBy { it.createdAt }.forEach { record ->
                        jsonArray.put(record.toJson())
                    }
                    editor.putString(dateKey, jsonArray.toString())
                }
                editor.apply()
                onComplete?.invoke()
            }
            .addOnFailureListener { error ->
                Log.e(TAG, "Firebase record sync failed: ${error.message}", error)
                onComplete?.invoke()
            }
    }

    fun getRecordsForDate(context: Context, dateKey: String): List<ConversationRecord> {
        return loadRecords(context, dateKey)
            .filterNot { it.isTrashed }
            .sortedBy { it.createdAt }
    }

    fun getRecordCountForDate(context: Context, dateKey: String): Int {
        return loadRecords(context, dateKey).count { !it.isTrashed }
    }

    fun getTodayDateKey(): String = formatDateKey(System.currentTimeMillis())

    fun toggleFavorite(
        context: Context,
        dateKey: String,
        createdAt: Long,
    ) {
        val records = loadRecords(context, dateKey).map { record ->
            if (record.createdAt == createdAt) {
                record.copy(isFavorite = !record.isFavorite)
            } else {
                record
            }
        }
        persistRecords(context, dateKey, records)
        records.firstOrNull { it.createdAt == createdAt }?.let { record ->
            updateRecordFlagsInFirebase(context, record)
        }
    }

    fun moveToTrash(
        context: Context,
        dateKey: String,
        createdAt: Long,
    ) {
        val records = loadRecords(context, dateKey).map { record ->
            if (record.createdAt == createdAt) {
                record.copy(isTrashed = true, isFavorite = false)
            } else {
                record
            }
        }
        persistRecords(context, dateKey, records)
        records.firstOrNull { it.createdAt == createdAt }?.let { record ->
            updateRecordFlagsInFirebase(context, record)
        }
    }

    fun permanentlyDelete(
        context: Context,
        dateKey: String,
        createdAt: Long,
    ) {
        val records = loadRecords(context, dateKey).filterNot { it.createdAt == createdAt }
        persistRecords(context, dateKey, records)
        deleteRecordFromFirebase(context, dateKey, createdAt)
    }

    fun restoreFromTrash(
        context: Context,
        dateKey: String,
        createdAt: Long,
        onComplete: (() -> Unit)? = null,
    ) {
        val records = loadRecords(context, dateKey).map { record ->
            if (record.createdAt == createdAt) {
                record.copy(isTrashed = false)
            } else {
                record
            }
        }
        persistRecords(context, dateKey, records)
        records.firstOrNull { it.createdAt == createdAt }?.let { record ->
            updateRecordFlagsInFirebase(context, record, onComplete)
        } ?: onComplete?.invoke()
    }

    fun getTrashedRecords(context: Context): List<ConversationRecord> {
        return prefs(context).all.keys
            .sorted()
            .flatMap { key -> loadRecords(context, key) }
            .filter { it.isTrashed }
            .sortedByDescending { it.createdAt }
    }

    fun findRecord(
        context: Context,
        dateKey: String,
        createdAt: Long,
    ): ConversationRecord? {
        return loadRecords(context, dateKey).firstOrNull { it.createdAt == createdAt }
    }

    fun getFavoriteRecords(context: Context, limit: Int? = 3): List<ConversationRecord> {
        val favorites = prefs(context).all.keys
            .sorted()
            .flatMap { key -> loadRecords(context, key) }
            .filter { it.isFavorite && !it.isTrashed }
            .sortedByDescending { it.createdAt }

        return if (limit == null) {
            favorites
        } else {
            favorites.take(limit)
        }
    }

    fun getAllActiveRecords(context: Context): List<ConversationRecord> {
        return prefs(context).all.keys
            .sorted()
            .flatMap { key -> loadRecords(context, key) }
            .filterNot { it.isTrashed }
            .sortedByDescending { it.createdAt }
    }

    fun updateRecordTitle(
        context: Context,
        dateKey: String,
        createdAt: Long,
        title: String,
        onComplete: (() -> Unit)? = null,
    ) {
        val trimmedTitle = title.trim()
        if (trimmedTitle.isBlank()) {
            onComplete?.invoke()
            return
        }

        val updatedRecords = loadRecords(context, dateKey).map { record ->
            if (record.createdAt == createdAt) record.copy(title = trimmedTitle) else record
        }
        persistRecords(context, dateKey, updatedRecords)

        updatedRecords.firstOrNull { it.createdAt == createdAt }?.let { record ->
            updateRecordContentInFirebase(context, record, onComplete)
        } ?: onComplete?.invoke()
    }

    fun getTodayLabel(): String {
        val formatter = SimpleDateFormat("EEEE d", Locale.ENGLISH)
        return formatter.format(Date())
    }

    fun getCurrentMonthLabel(): String {
        val formatter = SimpleDateFormat("MMMM yyyy", Locale.ENGLISH)
        return formatter.format(Date())
    }

    fun getWeekDates(): List<Calendar> {
        val today = Calendar.getInstance()
        val calendars = mutableListOf<Calendar>()
        for (offset in -3..3) {
            val clone = today.clone() as Calendar
            clone.add(Calendar.DAY_OF_MONTH, offset)
            calendars.add(clone)
        }
        return calendars
    }

    private fun loadRecords(context: Context, dateKey: String): List<ConversationRecord> {
        val raw = prefs(context).getString(dateKey, "[]").orEmpty()
        val jsonArray = JSONArray(raw)
        val result = mutableListOf<ConversationRecord>()
        for (index in 0 until jsonArray.length()) {
            val item = jsonArray.optJSONObject(index) ?: continue
            result.add(item.toConversationRecord(dateKey))
        }
        return result
    }

    private fun persistRecords(
        context: Context,
        dateKey: String,
        records: List<ConversationRecord>,
    ) {
        val jsonArray = JSONArray()
        records.forEach { record ->
            jsonArray.put(record.toJson())
        }

        prefs(context)
            .edit()
            .putString(dateKey, jsonArray.toString())
            .apply()
    }

    private fun saveRecordToFirebase(
        context: Context,
        record: ConversationRecord,
        startedAt: Long,
        endedAt: Long,
    ) {
        val uid = currentUserUid(context) ?: return
        val data = mapOf(
            "title" to record.title,
            "subtitle" to record.subtitle,
            "dateKey" to record.dateKey,
            "timeRange" to record.timeRange,
            "createdAt" to record.createdAt,
            "startedAt" to startedAt,
            "endedAt" to endedAt,
            "isFavorite" to record.isFavorite,
            "selfSpeech" to record.selfSpeech,
            "selfSummary" to record.selfSummary,
            "otherSpeech" to record.otherSpeech,
            "otherSummary" to record.otherSummary,
            "isTrashed" to record.isTrashed,
        )

        FirebaseFirestore.getInstance()
            .collection("users")
            .document(uid)
            .collection("records")
            .document(record.firebaseDocumentId())
            .set(data)
            .addOnFailureListener { error ->
                Log.e(TAG, "Firebase record save failed: ${error.message}", error)
            }
    }

    private fun updateRecordFlagsInFirebase(
        context: Context,
        record: ConversationRecord,
        onComplete: (() -> Unit)? = null,
    ) {
        val uid = currentUserUid(context) ?: return

        FirebaseFirestore.getInstance()
            .collection("users")
            .document(uid)
            .collection("records")
            .document(record.firebaseDocumentId())
            .update(
                mapOf(
                    "isFavorite" to record.isFavorite,
                    "isTrashed" to record.isTrashed,
                ),
            )
            .addOnSuccessListener {
                onComplete?.invoke()
            }
            .addOnFailureListener { error ->
                Log.e(TAG, "Firebase record flag update failed: ${error.message}", error)
                onComplete?.invoke()
            }
    }

    private fun updateRecordContentInFirebase(
        context: Context,
        record: ConversationRecord,
        onComplete: (() -> Unit)? = null,
    ) {
        val uid = currentUserUid(context) ?: run {
            onComplete?.invoke()
            return
        }

        FirebaseFirestore.getInstance()
            .collection("users")
            .document(uid)
            .collection("records")
            .document(record.firebaseDocumentId())
            .update("title", record.title)
            .addOnSuccessListener {
                onComplete?.invoke()
            }
            .addOnFailureListener { error ->
                Log.e(TAG, "Firebase record title update failed: ${error.message}", error)
                onComplete?.invoke()
            }
    }

    private fun deleteRecordFromFirebase(context: Context, dateKey: String, createdAt: Long) {
        val uid = currentUserUid(context) ?: return

        FirebaseFirestore.getInstance()
            .collection("users")
            .document(uid)
            .collection("records")
            .document("${dateKey}_${createdAt}")
            .delete()
            .addOnFailureListener { error ->
                Log.e(TAG, "Firebase record delete failed: ${error.message}", error)
            }
    }

    private fun currentUserUid(context: Context): String? {
        if (FirebaseApp.getApps(context).isEmpty()) return null
        return FirebaseAuth.getInstance().currentUser?.uid
    }

    private fun prefs(context: Context) =
        context.getSharedPreferences("${PREFS_NAME}_${currentUserUid(context) ?: "guest"}", Context.MODE_PRIVATE)

    private fun ConversationRecord.firebaseDocumentId(): String = "${dateKey}_${createdAt}"

    private fun ConversationRecord.toJson(): JSONObject =
        JSONObject().apply {
            put("title", title)
            put("subtitle", subtitle)
            put("dateKey", dateKey)
            put("timeRange", timeRange)
            put("startedAt", startedAt)
            put("createdAt", createdAt)
            put("isFavorite", isFavorite)
            put("selfSpeech", selfSpeech)
            put("selfSummary", selfSummary)
            put("otherSpeech", otherSpeech)
            put("otherSummary", otherSummary)
            put("isTrashed", isTrashed)
        }

    private fun JSONObject.toConversationRecord(defaultDateKey: String = optString("dateKey")): ConversationRecord =
        ConversationRecord(
            title = optString("title"),
            subtitle = optString("subtitle"),
            dateKey = optString("dateKey", defaultDateKey),
            timeRange = optString("timeRange"),
            startedAt = optLong("startedAt", optLong("createdAt")),
            createdAt = optLong("createdAt"),
            isFavorite = optBoolean("isFavorite", false),
            selfSpeech = optString("selfSpeech"),
            selfSummary = optString("selfSummary"),
            otherSpeech = optString("otherSpeech"),
            otherSummary = optString("otherSummary"),
            isTrashed = optBoolean("isTrashed", false),
        )

    private fun DocumentSnapshot.toConversationRecord(): ConversationRecord? {
        val createdAt = getLong("createdAt") ?: return null
        val dateKey = getString("dateKey").orEmpty()
        return ConversationRecord(
            title = getString("title").orEmpty(),
            subtitle = getString("subtitle").orEmpty(),
            dateKey = dateKey,
            timeRange = getString("timeRange").orEmpty(),
            startedAt = getLong("startedAt") ?: createdAt,
            createdAt = createdAt,
            isFavorite = getBoolean("isFavorite") ?: false,
            selfSpeech = getString("selfSpeech").orEmpty(),
            selfSummary = getString("selfSummary").orEmpty(),
            otherSpeech = getString("otherSpeech").orEmpty(),
            otherSummary = getString("otherSummary").orEmpty(),
            isTrashed = getBoolean("isTrashed") ?: false,
        )
    }

    private fun buildTimeRange(startedAt: Long, endedAt: Long): String {
        val formatter = SimpleDateFormat("hh:mm a", Locale.ENGLISH)
        return "${formatter.format(Date(startedAt))} - ${formatter.format(Date(endedAt))}".lowercase(Locale.ENGLISH)
    }

    private fun formatDateKey(timeMillis: Long): String {
        val formatter = SimpleDateFormat("yyyy-MM-dd", Locale.KOREA)
        return formatter.format(Date(timeMillis))
    }
}
