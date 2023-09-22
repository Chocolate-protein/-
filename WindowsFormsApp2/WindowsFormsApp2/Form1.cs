using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowsFormsApp2
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (comboBox1.SelectedItem.ToString() == "멜론") {
                label2.Text = comboBox1.SelectedItem.ToString();
                pictureBox1.ImageLocation = @"C:\Users\user\Desktop\WindowsFormsApp2\images\멜론.jpg";
            }
            else if (comboBox1.SelectedItem.ToString() == "복숭아")
            {
                label2.Text = comboBox1.SelectedItem.ToString();
                pictureBox1.ImageLocation = @"C:\Users\user\Desktop\WindowsFormsApp2\images\복숭아.jpg";
            }
            else if (comboBox1.SelectedItem.ToString() == "망고")
            {
                label2.Text = comboBox1.SelectedItem.ToString();
                pictureBox1.ImageLocation = @"C:\Users\user\Desktop\WindowsFormsApp2\images\망고.jpg";
            }
            else if (comboBox1.SelectedItem.ToString() == "딸기")
            {
                label2.Text = comboBox1.SelectedItem.ToString();
                pictureBox1.ImageLocation = @"C:\Users\user\Desktop\WindowsFormsApp2\images\딸기.jpg";
            }
            else if (comboBox1.SelectedItem.ToString() == "바나나")
            {
                label2.Text = comboBox1.SelectedItem.ToString();
                pictureBox1.ImageLocation = @"C:\Users\user\Desktop\WindowsFormsApp2\images\바나나.jpg";
            }
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            if(checkBox1.Checked == true)
            {
                checkBox2.Checked = false;
                checkBox3.Checked = false;
                label2.Text = "멜론";
                pictureBox1.ImageLocation = @"C:\Users\user\Desktop\WindowsFormsApp2\images\멜론.jpg";
            }
        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox2.Checked == true)
            {
                checkBox1.Checked = false;
                checkBox3.Checked = false;
                label2.Text = "복숭아";
                pictureBox1.ImageLocation = @"C:\Users\user\Desktop\WindowsFormsApp2\images\복숭아.jpg";
            }
        }

        private void checkBox3_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox3.Checked == true)
            {
                checkBox1.Checked = false;
                checkBox2.Checked = false;
                label2.Text = "망고";
                pictureBox1.ImageLocation = @"C:\Users\user\Desktop\WindowsFormsApp2\images\망고.jpg";
            }
        }
    }
}
