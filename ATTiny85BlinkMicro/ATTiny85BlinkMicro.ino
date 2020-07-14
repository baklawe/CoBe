#define MaxL 255
#define MinL 100
#define flash_D 33.333 //About 30 fps (delay[ms])
#define NUM_BITS 15
//int code_word = 335;
//PWM pins are 3, 5, 6, 9, 10, 11
//int pin15855 = 6;
//int pin2047 = 5;
//int pin51 = 3;
int code7[NUM_BITS]={MinL,MinL,MinL,MinL,MinL,MinL,MinL,MinL,MinL,MinL,MinL,MinL,MaxL,MaxL,MaxL};//000000000000111
int code51[NUM_BITS]={MinL,MinL,MinL,MinL,MinL,MinL,MinL,MinL,MinL,MaxL,MaxL,MinL,MinL,MaxL,MaxL};//000000000110011
int code85[NUM_BITS]={MinL,MinL,MinL,MinL,MinL,MinL,MinL,MinL,MaxL,MinL,MaxL,MinL,MaxL,MinL,MaxL};//000000001010101
int code127[NUM_BITS]={MinL,MinL,MinL,MinL,MinL,MinL,MinL,MinL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL};//000000001111111
int code265[NUM_BITS]={MinL,MinL,MinL,MinL,MinL,MinL,MaxL,MinL,MinL,MinL,MinL,MaxL,MinL,MinL,MaxL};//000000100001001
int code335[NUM_BITS]={MinL,MinL,MinL,MinL,MinL,MinL,MaxL,MinL,MaxL,MinL,MinL,MaxL,MaxL,MaxL,MaxL};//000000101001111
int code467[NUM_BITS]={MinL,MinL,MinL,MinL,MinL,MinL,MaxL,MaxL,MaxL,MinL,MaxL,MinL,MinL,MaxL,MaxL};//000000111010011
int code741[NUM_BITS]={MinL,MinL,MinL,MinL,MinL,MaxL,MinL,MaxL,MaxL,MaxL,MinL,MinL,MaxL,MinL,MaxL};//000001011100101
int code959[NUM_BITS]={MinL,MinL,MinL,MinL,MinL,MaxL,MaxL,MaxL,MinL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL};//000001110111111
int code1207[NUM_BITS]={MinL,MinL,MinL,MinL,MaxL,MinL,MinL,MaxL,MinL,MaxL,MaxL,MinL,MaxL,MaxL,MaxL};//000010010110111
int code1415[NUM_BITS]={MinL,MinL,MinL,MinL,MaxL,MinL,MaxL,MaxL,MinL,MinL,MinL,MinL,MaxL,MaxL,MaxL};//000010110000111
int code1583[NUM_BITS]={MinL,MinL,MinL,MinL,MaxL,MaxL,MinL,MinL,MinL,MaxL,MinL,MaxL,MaxL,MaxL,MaxL};//000011000101111
int code1753[NUM_BITS]={MinL,MinL,MinL,MinL,MaxL,MaxL,MinL,MaxL,MaxL,MinL,MaxL,MaxL,MinL,MinL,MaxL};//000011011011001
int code2007[NUM_BITS]={MinL,MinL,MinL,MinL,MaxL,MaxL,MaxL,MaxL,MaxL,MinL,MaxL,MinL,MaxL,MaxL,MaxL};//000011111010111
int code2211[NUM_BITS]={MinL,MinL,MinL,MaxL,MinL,MinL,MinL,MaxL,MinL,MaxL,MinL,MinL,MinL,MaxL,MaxL};//000100010100011
int code2293[NUM_BITS]={MinL,MinL,MinL,MaxL,MinL,MinL,MinL,MaxL,MaxL,MaxL,MaxL,MinL,MaxL,MinL,MaxL};//000100011110101
int code2733[NUM_BITS]={MinL,MinL,MinL,MaxL,MinL,MaxL,MinL,MaxL,MinL,MaxL,MinL,MaxL,MaxL,MinL,MaxL};//000101010101101
int code2875[NUM_BITS]={MinL,MinL,MinL,MaxL,MinL,MaxL,MaxL,MinL,MinL,MaxL,MaxL,MaxL,MinL,MaxL,MaxL};//000101100111011
int code3327[NUM_BITS]={MinL,MinL,MinL,MaxL,MaxL,MinL,MinL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL};//000110011111111
int code4685[NUM_BITS]={MinL,MinL,MaxL,MinL,MinL,MaxL,MinL,MinL,MaxL,MinL,MinL,MaxL,MaxL,MinL,MaxL};//001001001001101
int code5503[NUM_BITS]={MinL,MinL,MaxL,MinL,MaxL,MinL,MaxL,MinL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL};//001010101111111
int code6991[NUM_BITS]={MinL,MinL,MaxL,MaxL,MinL,MaxL,MaxL,MinL,MaxL,MinL,MinL,MaxL,MaxL,MaxL,MaxL};//001101101001111
int code8127[NUM_BITS]={MinL,MinL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MinL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL};//001111110111111
int code11191[NUM_BITS]={MinL,MaxL,MinL,MaxL,MinL,MaxL,MaxL,MaxL,MinL,MaxL,MaxL,MinL,MaxL,MaxL,MaxL};//010101110110111
int code15855[NUM_BITS]={MinL,MaxL,MaxL,MaxL,MaxL,MinL,MaxL,MaxL,MaxL,MaxL,MinL,MaxL,MaxL,MaxL,MaxL};//011110111101111
int code32767[NUM_BITS]={MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL,MaxL};//111111111111111

void setup() {
  // initialize digital pin LED_BUILTIN as an output.
//pinMode(pin51, OUTPUT);
}

// the loop function runs over and over again forever
void loop() {
      for (int j=0 ; j<NUM_BITS ;j++)
          {
            analogWrite(2,code51[j]);
            delay(flash_D); 
          } 

}
