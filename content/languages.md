Title: Hello, World! What Languages to Learn in 2019? (Altair visualizations)
Date: 2019-04-01 22:35
Tags: python languages altair visualization
Slug: languages

It's been three weeks into my Data Science immersive bootcamp and I'm having a blast coding again! I'm starting to get used to Python and new libraries (pandas, altair, etc.). If you're like me and you've previously programmed in other languages like C/Java, when learning any new language, there was some unlearning and relearning to do. In particular, I found myself consistently trying to reach for a for-from-until-increment-loop in Python, which doesn't exist!

While I'm getting the hang of Python, I'm starting to appreciate the ease of use, code readability, and structure of the language. This brought me to something I was curious about... What are the top programming languages today? Am I learning the "right" language? How about spoken languages? What will strengthen my portfolio?

Like anything else, what do you do when questions arise? I googled it! This is not meant to tell you what languages to pick up or to drop but mainly to fulfill my curiosity. *No extensive research was done during this sitting*. I did take some basic data though and played around with Altair visualizations... so read on if you'd like to learn some tips and tricks for visualization.

[(Interactive) Multi-series Line Chart - with color and chart size modification](#multi_series_line_chart)

[(Interactive) Horizontal Stacked Bar Chart - with tooltip, selection, and filtering](#stacked_bar_chart)

## Programming Languages

***

How many programming languages exist? There doesn't seem to be a definite answer amongst different sources I looked at... let's just say a lot (maybe somewhere between 500 and 2000). There are two existing indexes that I found tracking programming language popularity.

1. [TIOBE Index](https://www.tiobe.com/tiobe-index/) - based on number of search engine results for queries containing the name of the language
2. [PYPL - PopularitY of Programming Language](http://pypl.github.io/PYPL.html) - based on how often language tutorials are searched on Google

I tried using Google Trends myself, on programming languages that I've learned and used in the past. *Retrieved 01-Apr-2019.*

### Retrieve Data
At [Google Trends](https://trends.google.com/trends/?geo=US), I searched the following 5 different languages for comparison worldwide from 2004 to now, and downloaded the data for interest over time. 

<img src="images/google_trends.png" alt="google_trends" width="900">

#### Import libraries and data


```python
import pandas as pd
import altair as alt
alt.renderers.enable('notebook')
```




    RendererRegistry.enable('notebook')




```python
# header that I want is on row 2
interest_over_time = pd.read_csv('interest_over_time.csv', header = 1)
```

Google trends does not return absolute search counts. Instead, it gives a number to compare popularity between search terms. I'm referring to this as `relative_popularity`. This is achieved by the number of searches (for any search term) divided by the total number of searches on Google, for the chosen geography and time range, then scaled (0 to 100) based on the all search terms. For example, Java (with 100 in 2004-03) was the most popular between all five languages worldwide from 2004-present. All other values can be treated relative to that. 


```python
interest_over_time.head() # peek into data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Python: (Worldwide)</th>
      <th>Java: (Worldwide)</th>
      <th>C: (Worldwide)</th>
      <th>MATLAB: (Worldwide)</th>
      <th>Assembly language: (Worldwide)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004-01</td>
      <td>6</td>
      <td>93</td>
      <td>19</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004-02</td>
      <td>7</td>
      <td>98</td>
      <td>21</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2004-03</td>
      <td>7</td>
      <td>100</td>
      <td>20</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004-04</td>
      <td>6</td>
      <td>97</td>
      <td>20</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004-05</td>
      <td>6</td>
      <td>93</td>
      <td>19</td>
      <td>7</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### Visualize with Altair
[Altair](https://altair-viz.github.io/) is a visualization library for Python, based on Vega and Vega-Lite. First, I used `pd.melt()` to unpivot the data from wide to long format.


```python
long = pd.melt(interest_over_time, id_vars = ['Month'], 
               var_name = 'language', value_name = 'relative_popularity')
```


```python
long.head() # peek into data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>language</th>
      <th>relative_popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004-01</td>
      <td>Python: (Worldwide)</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004-02</td>
      <td>Python: (Worldwide)</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2004-03</td>
      <td>Python: (Worldwide)</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004-04</td>
      <td>Python: (Worldwide)</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004-05</td>
      <td>Python: (Worldwide)</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



Here's some simple Altair code to generate a multi-series line chart:


```python
alt.Chart(long).mark_line().encode(
    x = 'Month',
    y = 'relative_popularity',
    color = 'language' # separate line for each language
)
```


    <vega.vegalite.VegaLite at 0x11b929978>





    




![png](images/languages_19_2.png)


<a id='multi_series_line_chart'></a>
#### Tips and Tricks (make it easier to see!)
That's a bit hard to see on one page... we can play around with the following:

- **Date format on x-axis:** Change the data type of column `Month` to `datetime64[ns]` using `pd.to_datetime()` to reduce the noise on the x-axis.


```python
long['Month'] = pd.to_datetime(long['Month'])
```

- **Color:** Change the color scheme using `alt.Color` and `alt.Scale` if you think the blue, teal, green blends together in the chart above. Would you consider teal more green or blue? Other Vega color schemes can be found [here](https://vega.github.io/vega/docs/schemes/#reference).
- **Size:** Modify the chart size with `properties`.
- **Scaling:** Make chart axes scales `interactive` if you want to zoom into the bottom right noisier area.


```python
alt.Chart(long).mark_line().encode(
    x = 'Month',
    y = 'relative_popularity',
    color = alt.Color('language', 
                      scale = alt.Scale(scheme = 'set1')
                     )
).properties(
    width = 700, 
    height = 300
).interactive()
```


    <vega.vegalite.VegaLite at 0x11b9a0fd0>





    




![png](images/languages_24_2.png)


## Spoken Languages

***

On the other hand, there are **7,111 living spoken languages** that we know of to date according to [Ethnologue 2019, 22nd edition](https://www.ethnologue.com/)! I am supposedly trilingual but admittedly don't keep up the practice! I always wonder if I should be keeping up with these spoken languages... Spoken (or programming) languages are constantly evolving and their popularity shaped by the people and technolgies around the world.

I manually extracted the top 5 languages based on the number of `l1_speakers` (first language), `l2_speakers` (second language), and `total` number of speakers to play around with some more visualizations. 

Source: [Summary by language size](https://www.ethnologue.com/statistics/size), Ethnologue. *Retrieved 01-Apr-2019.*

### Retrieve Data


```python
spoken = pd.read_csv('spoken_language_popularity.csv')
```


```python
spoken # peak into data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>language</th>
      <th>l1_speakers</th>
      <th>l2_speakers</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>English</td>
      <td>379,007,140</td>
      <td>753,359,540</td>
      <td>1,132,366,680</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mandarin Chinese</td>
      <td>917,868,640</td>
      <td>198,728,000</td>
      <td>1,116,596,640</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hindi</td>
      <td>341,208,640</td>
      <td>274,266,900</td>
      <td>615,475,540</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spanish</td>
      <td>460,093,030</td>
      <td>74,242,700</td>
      <td>534,335,730</td>
    </tr>
    <tr>
      <th>4</th>
      <td>French</td>
      <td>77,177,210</td>
      <td>202,644,720</td>
      <td>279,821,930</td>
    </tr>
  </tbody>
</table>
</div>




```python
spoken.dtypes # check for column data types
```




    language       object
    l1_speakers    object
    l2_speakers    object
    total          object
    dtype: object



Since all column data types are `object`, numeric columns were converted to `float64` (units, in millions) by:

- **Removing commas**: in the numeric `str` values by using `str.replace()`
- **Converting dtype** `object` to `float` using `astype()`
- **Unit conversion:** Dividing by 1,000,000 to convert to units (in millions), and flooring the number by dividing by 1


```python
cols = ['l1_speakers', 'l2_speakers', 'total']
for col in cols:
    spoken[col] = spoken[col].map(lambda x: str(x).replace(',','')).astype(float) / 1_000_000 // 1
```


```python
spoken
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>language</th>
      <th>l1_speakers</th>
      <th>l2_speakers</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>English</td>
      <td>379.0</td>
      <td>753.0</td>
      <td>1132.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mandarin Chinese</td>
      <td>917.0</td>
      <td>198.0</td>
      <td>1116.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hindi</td>
      <td>341.0</td>
      <td>274.0</td>
      <td>615.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spanish</td>
      <td>460.0</td>
      <td>74.0</td>
      <td>534.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>French</td>
      <td>77.0</td>
      <td>202.0</td>
      <td>279.0</td>
    </tr>
  </tbody>
</table>
</div>



### Visualize with Altair

Here I converted the data to a long format as was done for the Programming languages. A separate data frame is also created to keep hold of the `total` values. 


```python
l1_l2 = pd.melt(spoken, id_vars = ['language'], value_vars = ['l1_speakers', 'l2_speakers'])
total = spoken[['language', 'total']]
```

Here's some simple Altair code to generate a horizontal bar chart:


```python
alt.Chart(l1_l2).mark_bar().encode(
    x = alt.X('sum(value)',
              axis = alt.Axis(title = 'Numbers of Speakers (in millions)'),
             ),
    y = alt.Y('language',
              axis = alt.Axis(title = 'Language'),
              sort = alt.EncodingSortField(op = 'count') # sort languages by total 
             ),
    color = 'variable',
    order = alt.Order('variable', sort = 'ascending') # put l1_speakers left of l2_speakers
).properties(
    width = 500,
    height = 200
)
```


    <vega.vegalite.VegaLite at 0x11b913780>





    




![png](images/languages_38_2.png)


<a id='stacked_bar_chart'></a>
#### Tips and Tricks (make it more interactive!) 
There are a couple of things we may want to do to try to make it more fun:

- **Selection:** Add a selection, which captures interactions from mouse clicks.


```python
selection = alt.selection_single(fields = ['variable'])
```

- **Legend:** Create a separate legend out of `mark_point()` and using it as radio buttons to act as inputs to `selection`.


```python
legend = alt.Chart(l1_l2).mark_point().encode(
    y = alt.Y('variable:N',
              axis = alt.Axis(title = None, orient = 'right')
             ),
    color = alt.Color('variable', legend = None)
).add_selection(
    selection
)
```

- **Filter:** Add `transform_filter` to the bar chart code, to transform the data based on `selection`
- **Tooltip:** Add `tooltip` to display details upon mouseover on different bars


```python
l1_l2_bar = alt.Chart(l1_l2).mark_bar().encode(
    x = alt.X('sum(value)',
              axis = alt.Axis(title = 'Numbers of Speakers (in millions)')
             ),
    y = alt.Y('language',
              axis = alt.Axis(title = 'Language'),
              sort = alt.EncodingSortField(op = 'count')
             ),
    color = alt.Color('variable', legend = None),
    order = alt.Order('variable', sort = 'ascending'), 
    tooltip = ['language','variable', 'value']
).transform_filter(
    selection
).properties(
    width = 500,
    height = 200
)
```

- **Bar Background:** Add a background bar chart, to capture the `total` number of speakers and using `alt.layer()` *(`+` can also be used)* and `alt.hconcat()` *(`|` can also be used)* to combine charts together.


```python
# background bars for total number of speakers
total_bar = alt.Chart(total).mark_bar(color = 'lightgray').encode(
    x = alt.X('total',
              axis = alt.Axis(title = 'Numbers of Speakers (in millions)'),
             ),
    y = alt.Y('language',
              axis = alt.Axis(title = 'Language'),
              sort = alt.EncodingSortField(op = 'count')
             ),
    tooltip = ['language','total']
).properties(
    width = 500,
    height = 200
)
```


```python
# compound charts 
total_bar + l1_l2_bar | legend 
```


    <vega.vegalite.VegaLite at 0x11a804198>





    




![png](images/languages_48_2.png)


You can try clicking the radio buttons on the legend and see how the chart changes! I'm still fairly new to Altair and all, so if there are better ways of doing what I tried, I'm open to feedback! Altair is also relatively new and consistently being updated so I'm sure that soon enough, there may be more concise ways of achieving the same charts.

**So... any thoughts on new languages to learn this year?**