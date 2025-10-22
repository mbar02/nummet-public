#include <cstdint>
#include <iostream>

#include "progressBar.h"

asyncProgressBar::asyncProgressBar(uint64_t samples, uint32_t columns, std::ostream *ostr) {
  this->columns = columns;
  this->steps   = samples;
  this->lastLim = 0xffffffff;
  this->status  = 0;
  this->ostr    = ostr;

  *(this->ostr) << std::endl << std::endl;

  this->update(0);
}

void asyncProgressBar::update(uint64_t step) {
  this->status += step;

  if(this->status > this->steps)
    this->status = this->steps;

  uint32_t lim = (uint32_t)( ( this->status * ( this->columns-2 ) ) / this->steps );

  if(lim > this->columns - 2 ) {
    *(this->ostr) << "Mhh, I got an overflow in ProgressBar :S";
    lim = this->columns - 2;
  }

  if(lim == this->lastLim) return;

  this->lastLim = lim;

  *(this->ostr) << "\r\033[2A";
  uint32_t i;
  *(this->ostr) << "<";
  for(i = 0; i<lim; i++)        *(this->ostr) << "#";
  for(; i<this->columns-2; i++) *(this->ostr) << ".";
  *(this->ostr) << ">" << std::endl;

  *(this->ostr) << "Simulazione completata al " << 100*this->status/this->steps << "%." << std::endl << std::flush;
  
}