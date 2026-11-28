@echo off
setlocal
rem ============================================================
rem �ϳ�һ�� ASR �طŲ�����Ƶ��Windows �Դ� SAPI�����߲���������
rem
rem �÷���
rem   make-test-audio.cmd                          ����� tests\assets\asr-test-question.wav
rem   make-test-audio.cmd D:\out\q.wav             ָ�����·��
rem   make-test-audio.cmd D:\out\q.wav "�Զ����İ�" ָ�����·�����İ�
rem
rem �����ʽ��16000 Hz / 16 bit / ������ PCM��
rem �� ASR ���������Ҫ��һ�£�app/audio/asr_replay.py����
rem
rem ע�⣺���ļ��������İ���cmd ��ϵͳ����ҳ������ϵͳΪ 936/GBK��
rem �����������У��뱣�ֱ��ļ�Ϊ ANSI��GBK�����뱣�档
rem ============================================================

rem ��Ŀ��Ŀ¼�����ű��� scripts\windows\ �£���������
for %%I in ("%~dp0..\..") do set "REPO_ROOT=%%~fI"

set "OUT_PATH=%~1"
if "%OUT_PATH%"=="" set "OUT_PATH=%REPO_ROOT%\tests\assets\asr-test-question.wav"

set "REPLAY_TEXT=%~2"
if "%REPLAY_TEXT%"=="" set "REPLAY_TEXT=����һ��ʲô��GDS"

rem ���Ŀ¼������ʱ�ȴ���
for %%I in ("%OUT_PATH%") do set "OUT_DIR=%%~dpI"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

rem �ϳɲ����������������� PowerShell������������ Unicode ���ݣ�
rem �����б������ִ� ASCII�������İ����ܴ���ҳӰ�죩��
rem SetOutputToWaveFile ֻд�ļ�����ţ�����ռ����������
set "REPLAY_OUT=%OUT_PATH%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); $synth.SetOutputToWaveFile($env:REPLAY_OUT, $format); $synth.Speak($env:REPLAY_TEXT); $synth.SetOutputToNull(); $synth.Dispose(); Write-Host ('Saved: ' + $env:REPLAY_OUT)"
if errorlevel 1 (
  echo �ϳ�ʧ�ܣ�PowerShell �˳��� %errorlevel%
  exit /b 1
)

echo �����ɻطŲ�����Ƶ��%OUT_PATH%
echo �İ���%REPLAY_TEXT%
exit /b 0