package com.example.myapplication

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.os.Bundle
import android.os.SystemClock
import android.text.Editable
import android.text.TextWatcher
import android.text.method.HideReturnsTransformationMethod
import android.text.method.PasswordTransformationMethod
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.myapplication.databinding.ActivitySignUpBinding
import com.google.firebase.auth.FirebaseAuthUserCollisionException
import java.io.File

class SignUpActivity : BaseActivity() {

    private lateinit var binding: ActivitySignUpBinding
    private var syncingAllTerms = false
    private var isRecording = false
    private var isPasswordConfirmVisible = false
    private var recorder: MediaRecorder? = null
    private var outputFile: File? = null

    private val requestMicPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            binding.switchMicPermission.isChecked = granted
            if (granted) {
                showVoicePopup()
            } else {
                Toast.makeText(this, getString(R.string.record_denied), Toast.LENGTH_LONG).show()
            }
            updateTermsState()
        }

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            binding.switchCameraPermission.isChecked = granted
            if (!granted) {
                Toast.makeText(this, getString(R.string.camera_denied), Toast.LENGTH_LONG).show()
            }
            updateTermsState()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySignUpBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.inputPasswordConfirm.transformationMethod = PasswordTransformationMethod.getInstance()
        binding.txtNicknameDuplicate.visibility = View.GONE
        binding.txtPasswordMismatch.visibility = View.GONE

        binding.inputName.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                binding.txtNicknameDuplicate.visibility = View.GONE
            }
            override fun afterTextChanged(s: Editable?) = Unit
        })

        val passwordWatcher = object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                updatePasswordMismatchState()
            }
            override fun afterTextChanged(s: Editable?) = Unit
        }
        binding.inputPassword.addTextChangedListener(passwordWatcher)
        binding.inputPasswordConfirm.addTextChangedListener(passwordWatcher)

        binding.btnPasswordConfirmVisibility.setOnClickListener {
            isPasswordConfirmVisible = !isPasswordConfirmVisible
            binding.inputPasswordConfirm.transformationMethod =
                if (isPasswordConfirmVisible) {
                    HideReturnsTransformationMethod.getInstance()
                } else {
                    PasswordTransformationMethod.getInstance()
                }
            binding.btnPasswordConfirmVisibility.alpha = if (isPasswordConfirmVisible) 1f else 0.55f
            binding.inputPasswordConfirm.setSelection(binding.inputPasswordConfirm.text?.length ?: 0)
        }

        binding.btnPasswordConfirmVisibility.alpha = 0.55f

        binding.termsHeader.setOnClickListener {
            toggleTermsSection()
        }
        renderTermsArrow(isExpanded = false)

        binding.switchAllTerms.setOnCheckedChangeListener { _, isChecked ->
            if (syncingAllTerms) return@setOnCheckedChangeListener

            if (binding.termsDetailWrap.visibility != View.VISIBLE) {
                binding.termsDetailWrap.visibility = View.VISIBLE
                renderTermsArrow(isExpanded = true)
            }

            if (isChecked) {
                requestMicIfNeeded()
                requestCameraIfNeeded()
            } else {
                binding.switchMicPermission.isChecked = false
                binding.switchCameraPermission.isChecked = false
                hideVoicePopup()
                updateTermsState()
            }
        }

        binding.switchMicPermission.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                requestMicIfNeeded()
            } else {
                hideVoicePopup()
                updateTermsState()
            }
        }

        binding.switchCameraPermission.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                requestCameraIfNeeded()
            } else {
                updateTermsState()
            }
        }

        binding.btnMicRecord.setOnClickListener {
            if (isRecording) {
                stopRecordingAndDismiss()
            } else {
                startRecording()
            }
        }

        binding.btnComplete.setOnClickListener {
            signUpWithFirebase()
        }
    }

    override fun onStop() {
        super.onStop()
        stopRecordingOnly()
    }

    private fun toggleTermsSection() {
        val expanded = binding.termsDetailWrap.visibility == View.VISIBLE
        binding.termsDetailWrap.visibility = if (expanded) View.GONE else View.VISIBLE
        renderTermsArrow(isExpanded = !expanded)
    }

    private fun renderTermsArrow(isExpanded: Boolean) {
        binding.imgTermsArrow.rotation = if (isExpanded) 90f else 0f
    }

    private fun requestMicIfNeeded() {
        if (hasPermission(Manifest.permission.RECORD_AUDIO)) {
            binding.switchMicPermission.isChecked = true
            showVoicePopup()
            updateTermsState()
        } else {
            requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun requestCameraIfNeeded() {
        if (hasPermission(Manifest.permission.CAMERA)) {
            binding.switchCameraPermission.isChecked = true
            updateTermsState()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun hasPermission(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
    }

    private fun updateTermsState() {
        val allGranted = binding.switchMicPermission.isChecked && binding.switchCameraPermission.isChecked
        if (binding.switchAllTerms.isChecked != allGranted) {
            syncingAllTerms = true
            binding.switchAllTerms.isChecked = allGranted
            syncingAllTerms = false
        }
        binding.btnComplete.isEnabled = allGranted && hasRecordedVoiceSample()
    }

    private fun hasRecordedVoiceSample(): Boolean {
        return outputFile?.exists() == true && outputFile?.length()?.let { it > 0L } == true && !isRecording
    }

    private fun showVoicePopup() {
        binding.voiceOverlay.visibility = View.VISIBLE
        binding.txtRecordTitle.setText(R.string.record_title)
        binding.imgMicIcon.setImageResource(R.drawable.ic_voice_mic)
    }

    private fun hideVoicePopup() {
        stopRecordingOnly()
        binding.voiceOverlay.visibility = View.GONE
    }

    private fun startRecording() {
        if (!hasPermission(Manifest.permission.RECORD_AUDIO)) {
            requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)
            return
        }

        outputFile = File(cacheDir, "signup_voice_note.m4a")

        recorder = MediaRecorder().apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            setOutputFile(outputFile?.absolutePath)
            prepare()
            start()
        }

        isRecording = true
        binding.txtOverlayTimer.base = SystemClock.elapsedRealtime()
        binding.txtOverlayTimer.start()
        binding.txtRecordTitle.setText(R.string.recording_now)
        binding.imgMicIcon.setImageResource(R.drawable.ic_voice_stop)
        updateTermsState()
    }

    private fun stopRecordingOnly() {
        if (!isRecording) return

        runCatching {
            recorder?.stop()
        }
        recorder?.release()
        recorder = null
        isRecording = false
        binding.txtOverlayTimer.stop()
        binding.txtOverlayTimer.base = SystemClock.elapsedRealtime()
        binding.txtRecordTitle.setText(R.string.record_title)
        binding.imgMicIcon.setImageResource(R.drawable.ic_voice_mic)
        updateTermsState()
    }

    private fun stopRecordingAndDismiss() {
        stopRecordingOnly()
        binding.voiceOverlay.visibility = View.GONE
    }

    private fun signUpWithFirebase() {
        binding.txtNicknameDuplicate.visibility = View.GONE
        updatePasswordMismatchState()

        val identifier = binding.inputName.text.toString().trim()
        val password = binding.inputPassword.text.toString()
        val confirmPassword = binding.inputPasswordConfirm.text.toString()

        when {
            identifier.isBlank() || password.isBlank() || confirmPassword.isBlank() -> {
                Toast.makeText(this, R.string.login_empty_fields, Toast.LENGTH_SHORT).show()
                return
            }
            password.length < 6 -> {
                Toast.makeText(this, R.string.signup_password_short, Toast.LENGTH_SHORT).show()
                return
            }
            password != confirmPassword -> {
                Toast.makeText(this, R.string.signup_password_mismatch, Toast.LENGTH_SHORT).show()
                return
            }
            !hasRecordedVoiceSample() -> {
                Toast.makeText(this, R.string.signup_voice_required, Toast.LENGTH_SHORT).show()
                if (binding.voiceOverlay.visibility != View.VISIBLE) {
                    showVoicePopup()
                }
                return
            }
        }

        FirebaseUserStore.signUp(
            context = this,
            identifier = identifier,
            password = password,
            voiceSampleFile = outputFile?.takeIf { it.exists() && it.length() > 0L },
            onSuccess = {
                hideVoicePopup()
                startActivity(Intent(this, HomeActivity::class.java))
                finish()
            },
            onFailure = { error ->
                if (error is FirebaseAuthUserCollisionException) {
                    binding.txtNicknameDuplicate.visibility = View.VISIBLE
                    return@signUp
                }

                Toast.makeText(
                    this,
                    getString(R.string.signup_failed, error.localizedMessage ?: error.message.orEmpty()),
                    Toast.LENGTH_LONG,
                ).show()
            },
        )
    }

    private fun updatePasswordMismatchState() {
        val password = binding.inputPassword.text?.toString().orEmpty()
        val confirmPassword = binding.inputPasswordConfirm.text?.toString().orEmpty()
        val shouldShowMismatch =
            password.isNotEmpty() &&
                confirmPassword.isNotEmpty() &&
                password != confirmPassword

        binding.txtPasswordMismatch.visibility = if (shouldShowMismatch) View.VISIBLE else View.GONE
    }
}
