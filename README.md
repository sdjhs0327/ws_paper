# ws_paper
논문제목: 구글트렌드를 활용한 주식시장 예측을 위한 진보된 인공신 경망 접근법

논문개요
진보된 인공신경망에 대한 선행연구 결과를 미국 ETF를 대상으로 구현 및 검증한다.
또한, 구글트렌드 데이터를 설명변수로 추가 활용했을 때 예측 성능의 개선 여부를 확인한다.
예측 대상 자산은 미국 금융시장에서 거래량 기준으로 활발히 거래되며, 글로벌 자산 배분 전략에도 가장 폭넓게 활용되는 SPY, VWO, VEA, TLT 등의 ETF를 대상한다.
주요 설명변수로는 글로벌 자산 배분 결정에 핵심적 역할을 하는 거시지표인 국채 장단기 금리 스프레드, CBOE VIX 지수, S&P PE Ratio 등을 활용한다.
구글트렌드 데이터는 Peris(2013), Challet(2014) 등이 제안한 키워드를 차용한다.
진보된 인공신경망은 Pang(2018)이 활용한 ELSTM 구조를 활용하여 미국 금융시장에 대해 구현 및 검증한다.
거시지표를 활용한 ELSTM의 예측결과와 구글트렌드 데이터를 추가 활용한 결과를 Win Ratio, Sharpe Ratio, IR, MDD 등의 지표를 활용하여 비교한다.

현 연구수준
Pang(2018)을 참고하여 ELSTM 모형 구조 구현, 구글트렌드 활용 성능개선 확인보완

연구방향
평가 지표를 개선하고, 모형 구조 진단을 위한 Test 기간 변경을 수행한다.

