package com.example.myapplication

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException

object OpenAiSummaryService {

    data class SummaryResult(
        val text: String? = null,
        val errorMessage: String? = null,
    )

    private val client = OkHttpClient()
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()
    private const val ENDPOINT_SUMMARIZE_CONVERSATION = "/summarize_conversation_record"

    fun summarizeConversation(
        baseUrl: String?,
        conversationText: String,
        onResult: (String?) -> Unit,
    ) {
        summarizeConversationDetailed(baseUrl, conversationText) { result ->
            onResult(result.text)
        }
    }

    fun summarizeConversationDetailed(
        baseUrl: String?,
        conversationText: String,
        onResult: (SummaryResult) -> Unit,
    ) {
        val normalizedBaseUrl = baseUrl?.trim()?.removeSuffix("/").orEmpty()
        if (normalizedBaseUrl.isBlank()) {
            onResult(SummaryResult(errorMessage = "Backend URL is not available."))
            return
        }

        if (conversationText.isBlank()) {
            onResult(SummaryResult(errorMessage = "Conversation text is empty."))
            return
        }

        val payload = JSONObject().apply {
            put("conversation_text", conversationText)
        }

        val request = Request.Builder()
            .url("$normalizedBaseUrl$ENDPOINT_SUMMARIZE_CONVERSATION")
            .addHeader("Content-Type", "application/json")
            .post(payload.toString().toRequestBody(jsonMediaType))
            .build()

        client.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: IOException) {
                onResult(SummaryResult(errorMessage = "Conversation summary request failed."))
            }

            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                response.use {
                    val body = it.body?.string().orEmpty()

                    if (!it.isSuccessful) {
                        onResult(SummaryResult(errorMessage = parseErrorMessage(it.code, body)))
                        return
                    }

                    val text = parseOutputText(body)?.trim()?.takeIf(String::isNotBlank)
                    if (text.isNullOrBlank()) {
                        onResult(SummaryResult(errorMessage = "Conversation summary is empty."))
                    } else {
                        onResult(SummaryResult(text = text))
                    }
                }
            }
        })
    }

    private fun parseOutputText(body: String): String? {
        return runCatching {
            val json = JSONObject(body)
            json.optString("title").ifBlank { null }
        }.getOrNull()
    }

    private fun parseErrorMessage(code: Int, body: String): String {
        val apiMessage = runCatching {
            val json = JSONObject(body)
            json.optString("detail").ifBlank {
                json.optString("error")
            }
        }.getOrNull().orEmpty()

        val mappedMessage = when (code) {
            404 -> "Summary endpoint is unavailable on the backend."
            429 -> "Summary request was rate limited."
            else -> "Backend summary request failed. (HTTP $code)"
        }

        return if (apiMessage.isBlank()) {
            mappedMessage
        } else {
            "$mappedMessage\n$apiMessage"
        }
    }
}
