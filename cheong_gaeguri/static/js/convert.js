var button = document.querySelector("#switch");
try {
    var recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition || window.mozSpeechRecognition || window.msSpeechRecognition)();
} catch(e){
    console.error(e);
}
function toggle(element) {
    if(element.checked == true){
        speech_to_text()
    }
    else if(element.checked == false){
        stop()
    }
    }
function speech_to_text(){
    recognition.start();
    isRecognizing = true;
    recognition.lang="ko-KR"
    recognition.onstart = function(){
        document.getElementById("mouth").value = "음성인식 시작..."
        // console.log("음성인식이 시작 되었습니다. 이제 마이크에 무슨 말이든 하세요.")
        // message.innerHTML = "음성인식 시작...";
        // button.innerHTML = "Listening...";
        // button.disabled = true;
    }
//    recognition.onspeechend = function(){
//         // message.innerHTML = "버튼을 누르고 아무말이나 하세요.";
//         // // button.disabled = false;
//         // button.innerHTML = "Start STT";
//     }
    recognition.onresult = function(event) {
        document.getElementById("mouth").value = event.results[0][0].transcript
        // console.log('You said: ', event.results[0][0].transcript);
        // 결과를 출력 
        var resText = event.results[0][0].transcript;
        // korea.innerHTML = resText;
        //text to sppech
        // text_to_speech(resText);
    };
    recognition.onend = function(){
        // message.innerHTML = "버튼을 누르고 아무말이나 하세요.";
        // button.disabled = false;
        // button.innerHTML = "Start STT";
        isRecognizing = false;
    }
}
function stop(){
    console.log("멈춰!")
    // console.log(res)
    recognition.stop();
    // message.innerHTML = "버튼을 누르고 아무말이나 하세요.";
    // button.disabled = true;
    // button.innerHTML = "Start STT";
    isRecognizing = false;
}
// Text to speech
function text_to_speech(txt){
    // Web Speech API - speech synthesis
    if ('speechSynthesis' in window) {
        // Synthesis support. Make your web apps talk!
            console.log("음성합성을 지원하는  브라우저입니다.");
    }
    var msg = new SpeechSynthesisUtterance();
    var voices = window.speechSynthesis.getVoices();
    //msg.voice = voices[10]; // 두번째 부터 완전 외국인 발음이 됨. 사용하지 말것.
    msg.voiceURI = 'native';
    msg.volume = 1; // 0 to 1
    msg.rate = 1.3; // 0.1 to 10
    //msg.pitch = 2; //0 to 2
    msg.text = txt;
    msg.lang = 'ko-KR';
    msg.onend = function(e) {
        if(isRecognizing == false){
            recognition.start();    
        }
            console.log('Finished in ' + event.elapsedTime + ' seconds.');
    };
    window.speechSynthesis.speak(msg);
}
// ----------------------------------------------------- 형태소 추출 스크립트(python 호출 필요)---------------------------------------------------
function get_keyword(sentence){
}
//--------------------------------------------------------애니메이션 호출 스크립트(python 호출 필요?)------------------------------------------
function get_animation(key_word){
    get_keyword(sentence)
}
//--------------------------------------------------------애니메이션 실행 스크립트------------------------------------------------------------------
function play(){
}