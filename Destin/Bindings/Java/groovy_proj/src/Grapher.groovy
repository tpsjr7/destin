/**
 * Created by tedsandersjr on 10/25/14.
 */
@Grab(group='jfree', module='jfreechart', version='1.0.5')
@Grab(group='jfree', module='jcommon', version='1.0.9')
public class UsedToExposeAnnotationToComplier {}
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.general.DefaultPieDataset
import org.jfree.data.xy.DefaultTableXYDataset
import org.jfree.data.xy.XYSeries
import groovy.swing.SwingBuilder
import java.awt.*
import javax.swing.WindowConstants as WC
class Charter {
    def xys = new XYSeries("some data",true, false)
    def chart
    def init() {
        def tableDS = new DefaultTableXYDataset()
        tableDS.addSeries(xys)

        chart = ChartFactory.createXYLineChart("Data", "X", "Y", tableDS, PlotOrientation.VERTICAL, false, false, false)
        chart.backgroundPaint = Color.white

        def swing = new SwingBuilder()
        def frame = swing.frame(title:'Groovy Chart', defaultCloseOperation:WC.HIDE_ON_CLOSE) {
            panel(id:'canvas') { widget(new ChartPanel(chart)) }
        }

        addExampleData()

        frame.pack()
        frame.show()
    }

    def addExampleData(){
        xys.add(1,1)
        xys.add(2,2)
        xys.add(3,1)
        xys.add(4,4)
    }
}

ct = new Charter()

ct.init()